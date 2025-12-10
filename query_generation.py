# ===========================================
# GENERAZIONE DOMANDE v3.3 — Copertura Completa Template + Rimozione Sinonimi
# - GBNF e Validazione per TUTTI i template (How, Why, State, etc.)
# - Rimossa logica 'SYN' (sinonimi) e 'lemma_tokens'
# - Overlap k=1 per tutti i template
# - Multi-candidato con rerank
# ===========================================

import os, json, re, csv, time, math
from typing import Dict, Any, List, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import numpy as np

print("--- SCRIPT INIZIATO CORRETTAMENTE (v3.3) ---")

# ---------- CONFIG ----------
SHARDS_DIR       = "shards/v2/annotations/shards_flat/"
OUT_DIR          = "shards_flat/annotations/gen_out"
AUDIT_CSV_DIR    = "shards_flat/annotations/gen_out_audit"

BATCH_SIZE       = 16
MAX_NEW_TOKENS   = 64
DELTA_T          = 1.0          # ampliata la finestra temporale per catturare narrations al bordo
PREVIEW_EVERY    = 100
MAX_INSIDE_LINES = 24           # maggior copertura di righe
N_TRIALS         = 5             # più candidati per rerank

PARALLEL_WORKERS = 16
OPENAI_BASE_URL  = "http://127.0.0.1:8080/v1"
OPENAI_API_KEY   = "sk-local"

# Sampling
TEMPERATURE        = 0.0
USE_TEMP_BURST     = True
TEMP_BURST         = 0.2

# Grammar constraints (llama-server)
USE_GRAMMAR = True  # attiva GBNF vincolante per i template

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(AUDIT_CSV_DIR, exist_ok=True)

# ---------- CLIENT LLM (OpenAI-compat) ----------
_client_mode = None
try:
    from openai import OpenAI
    _openai_client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    _client_mode = "openai"
except Exception:
    import requests
    _session = requests.Session()
    _session.headers.update({"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"})
    _client_mode = "requests"

def _chat_completion(prompt: str, max_tokens: int = MAX_NEW_TOKENS, seed: int = 42,
                     temperature: float = TEMPERATURE, timeout: float = 30.0,
                     grammar: str = "") -> str:
    messages = [
        {"role": "system", "content": "You are a strict question generator. Obey TEMPLATE and GENERAL RULES exactly. Start with the required first word."},
        {"role": "user", "content": prompt}
    ]
    if _client_mode == "openai":
        resp = _openai_client.chat.completions.create(
            model="local",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            seed=seed,
            n=1,
            stop=["\n\n"],
            extra_body=({"grammar": grammar} if grammar else None)
        )
        return (resp.choices[0].message.content or "").strip()
    else:
        payload = {
            "model": "local",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "seed": seed,
            "n": 1,
            "stop": ["\n\n"],
        }
        if grammar:
            payload["grammar"] = grammar
        r = _session.post(f"{OPENAI_BASE_URL}/chat/completions", data=json.dumps(payload), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

# ---------- IO helpers ----------
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield json.loads(line.replace("NaN","null").replace("Infinity","null").replace("-Infinity","null"))
                except:
                    continue

def append_jsonl(path: str, rows, do_flush: bool = True):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)); f.write("\n")
        if do_flush:
            f.flush()
            try: os.fsync(f.fileno())
            except: pass

# ---------- RESUME helpers ----------
def make_input_key(rec: Dict[str,Any]) -> Tuple:
    return (
        rec.get("video_uid"),
        rec.get("matched_template"),
        float(rec.get("query_start_sec", 0.0)),
        float(rec.get("query_end_sec", 0.0))
    )

def load_resume_sets(out_jsonl_path: str) -> Tuple[set, set]:
    seen_input_keys, seen_triplets = set(), set()
    if not os.path.exists(out_jsonl_path):
        return seen_input_keys, seen_triplets
    for row in read_jsonl(out_jsonl_path):
        try:
            seen_input_keys.add((
                row.get("video_uid"),
                row.get("template"),
                float(row.get("query_start_sec", 0.0)),
                float(row.get("query_end_sec", 0.0))
            ))
            qtxt = (row.get("query","") or "").strip()
            seen_triplets.add((row.get("video_uid"), row.get("template"), qtxt))
        except Exception:
            continue
    return seen_input_keys, seen_triplets

# ---------- Window utilities ----------
def window_lines(item: Dict[str,Any], delta: float = DELTA_T, max_lines: int = MAX_INSIDE_LINES) -> List[str]:
    qs = float(item["query_start_sec"]); qe = float(item["query_end_sec"])
    lines = []
    for n in (item.get("narrations") or []):
        ts  = n.get("timestamp_sec", None)
        txt = (n.get("narration_text") or "").strip()
        if not txt or ts is None:
            continue
        if (qs - delta) <= float(ts) <= (qe + delta):
            lines.append(txt)
    # de-dup semplice
    deduped, prev = [], None
    for s in lines:
        if s != prev:
            deduped.append(s); prev = s
    if len(deduped) > max_lines:
        idx = np.linspace(0, len(deduped)-1, max_lines).round().astype(int)
        deduped = [deduped[i] for i in idx]
    return deduped

# ---------- Prompt & cleaning ----------
NO_QUESTION_RE = re.compile(r'^\s*\(?no question can be generated from the given narrations\)?\s*\?$', re.I)

def _strip_prefixes(s: str) -> str:
    s = re.sub(r"^\s*(?:---\s*QUESTION\s*---|Question\s*:)\s*", "", s, flags=re.I).strip()
    s = re.sub(r"^\s*(?:-+\s*|\*+\s*)", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r'#\w+\b', '', s).strip()  # rimuove tag per il prompt
    return s

def normalize_question(text: str) -> str:
    q = _strip_prefixes(text or "")
    q = re.sub(r"[.!]+$", "", q).strip()
    if q and not q.endswith("?"):
        q += "?"
    return q

# ---------- TEMPLATE-SPECIFIC RULES ----------
# MODIFICA 1: Aggiunte regole per i template mancanti (Why, How, State)
TEMPLATE_RULES = {
    "Objects: Where is object X before / after event Y?": (
        "CRITICAL RULE for this template:\n"
        "- Your question MUST start with 'Where'.\n"
        "- You MUST include 'before' or 'after'.\n"
        "- Examples: 'Where is X before Y?', 'Where is X after Y?'\n"
    ),
    "Objects: What did I put in X?": (
        "CRITICAL RULE for this template:\n"
        "- Start with 'What'.\n"
        "- Use a placement verb: 'put', 'place', 'drop', 'insert', 'pour', 'throw'.\n"
        "- Include a containment/location preposition: 'in', 'into', 'inside', 'onto'.\n"
        "- Example: 'What did C put in the bowl?'\n"
    ),
    "Objects: In what location did I see object X ?": (
        "CRITICAL RULE for this template:\n"
        "- Ask about LOCATION/POSITION.\n"
        "- Start with: 'Where', 'In what location', 'On what surface', 'Inside what'.\n"
    ),
    "Objects: How many X's? (quantity question)": (
        "CRITICAL RULE for this template:\n"
        "- Start with 'How many' or 'How much'.\n"
        "- Ask about COUNT or QUANTITY.\n"
    ),
    "Objects: What X did I Y?": (
        "CRITICAL RULE for this template:\n"
        "- Start with 'What' and focus on the OBJECT.\n"
        "- Example: 'What does C cut?'\n"
    ),
    "Objects: Where is object X?": (
        "CRITICAL RULE for this template:\n"
        "- Start with 'Where' or 'In what location'.\n"
        "- Ask about LOCATION/POSITION.\n"
    ),
    "Place: Where did I put X?": (
        "CRITICAL RULE for this template:\n"
        "- Start with 'Where' and ask about SPATIAL LOCATION of the placement.\n"
    ),
    "Objects: What X is Y?": (
        "CRITICAL RULE for this template:\n"
        "- Ask about STATE/PROPERTY/ATTRIBUTE.\n"
        "- Start with 'What' (e.g., 'What color...', 'What state...').\n"
    ),
    "People: Who did I talk to in location X?": (
        "CRITICAL RULE for this template:\n"
        "- Start with 'Who' and ask about PEOPLE present/interacting.\n"
    ),
    "People: Who did I interact with when I did activity X?": (
        "CRITICAL RULE for this template:\n"
        "- Start with 'Who' and ask about PEOPLE involved.\n"
    ),
    "Objects: State of an object": (
        "CRITICAL RULE for this template:\n"
        "- Ask about STATE or CONDITION (open/closed, on/off, full/empty, etc.).\n"
        "- Start with 'What' (e.g., 'What is the state of...').\n"
    ),
    "Action: Why did I do action X?": (
        "CRITICAL RULE for this template:\n"
        "- Start with 'Why'.\n"
        "- Ask about the REASON or PURPOSE of an action.\n"
    ),
    "Action: How did I do action X?": (
        "CRITICAL RULE for this template:\n"
        "- Start with 'How'.\n"
        "- Ask about the MANNER or METHOD of an action.\n"
        "- Do NOT ask 'How many' or 'How much'.\n"
    ),
    "Action: What is the action that I am doing?": (
        "CRITICAL RULE for this template:\n"
        "- Start with 'What'.\n"
        "- Ask to identify the action itself.\n"
    ),
}

def get_template_rule(tpl: str) -> str:
    for tpl_key, rule in TEMPLATE_RULES.items():
        if tpl_key.lower() == tpl.lower():
            return rule
    # Fallback per template non mappati (se ce ne fossero)
    if "where" in tpl.lower(): return TEMPLATE_RULES["Objects: Where is object X?"]
    if "how many" in tpl.lower(): return TEMPLATE_RULES["Objects: How many X's? (quantity question)"]
    if "who" in tpl.lower(): return TEMPLATE_RULES["People: Who did I talk to in location X?"]
    if "why" in tpl.lower(): return TEMPLATE_RULES["Action: Why did I do action X?"]
    if "how" in tpl.lower(): return TEMPLATE_RULES["Action: How did I do action X?"]
    return "Follow the template semantics as closely as possible.\n"

# ---------- Few-shots ----------
# MODIFICA 2: Aggiunti few-shots per 'how' e 'why'
USE_FEWSHOTS = True
MAX_FEWSHOTS_PER_TPL = 2

FEWSHOTS = {
    "__generic__": [
        {"narr": ["He picks up a red mug", "He pours coffee into the mug"], "q": "What does he pick up?"},
    ],
    "before / after": [
        {"narr": ["The wall is white", "He paints the wall blue"], "q": "Where is the wall color after he paints it?"},
        {"narr": ["She opens the drawer", "She puts forks inside"], "q": "Where are the forks after she puts them in?"},
    ],
    "how many": [
        {"narr": ["He stacks three plates on the table"], "q": "How many plates does he stack?"},
    ],
    "how_action": [
        {"narr": ["He carefully slices the tomato"], "q": "How does he slice the tomato?"},
    ],
    "why_action": [
        {"narr": ["He picks up the keys to leave the house"], "q": "Why does he pick up the keys?"},
    ],
    "in what location": [
        {"narr": ["He puts the keys on the counter"], "q": "Where does he put the keys?"},
    ],
    "who": [
        {"narr": ["She talks to a man at the door"], "q": "Who does she talk to?"},
    ],
    "what-object": [
        {"narr": ["She cuts a cucumber with a knife"], "q": "What does she cut?"},
    ],
}

def fewshot_block_for_template(tpl: str, k: int = MAX_FEWSHOTS_PER_TPL) -> str:
    tl = (tpl or "").lower()
    if "before / after" in tl:
        key = "before / after"
    elif "how many" in tl or "quantity" in tl:
        key = "how many"
    elif "how did i" in tl:
        key = "how_action"
    elif "why did i" in tl:
        key = "why_action"
    elif "who did i" in tl or "people:" in tl:
        key = "who"
    elif "where did i put x" in tl or "in what location" in tl or "where is object x" in tl or "place:" in tl:
        key = "in what location"
    elif "what x did i y" in tl or "objects: what" in tl:
        key = "what-object"
    else:
        key = "__generic__" # 'State' e altri 'What' usano il generico
    exs = FEWSHOTS.get(key, FEWSHOTS["__generic__"])[:k]
    parts = []
    for i, ex in enumerate(exs, 1):
        narr = "\n".join(f"- {s}" for s in ex["narr"])
        parts.append(f"Example {i}:\nNarrations:\n{narr}\nQuestion:\n{ex['q']}\n")
    return "\n".join(parts)

# ---------- Tokenization & overlap (SENZA sinonimi) ----------
# MODIFICA 3: Rimossa logica SYN e canon(), rinominato _tokens
_wtok = re.compile(r"[A-Za-z0-9']+")

def _tokens(s: str) -> List[str]:
    # Non filtra più per lunghezza, include 'C', 'a', 'is' etc.
    return [t.lower() for t in _wtok.findall(s or "")]

def has_min_overlap(query: str, inside: List[str], k: int) -> bool:
    qset = set(_tokens(query))
    nset = set(_tokens(" ".join(inside)))
    return len(qset & nset) >= k

# ---------- BUILD PROMPT (con anchor tokens) ----------
def build_prompt_for_query(tpl: str, qs: float, qe: float, inside: List[str], max_lines: int = MAX_INSIDE_LINES) -> str:
    if not inside:
        inside = ["(no narration lines strictly inside the window)"]
    # rimuovi tag nel prompt (Unsure ignorato nei filtri)
    inside = [re.sub(r'#\w+\b', '', s or '').strip() for s in inside]
    if len(inside) > max_lines:
        idx = np.linspace(0, len(inside)-1, max_lines).round().astype(int)
        inside = [inside[i] for i in idx]

    # Anchor tokens dai testi del window
    toks = sorted(set(t for s in inside for t in _tokens(s)))[:10] # Usa _tokens
    anchor = ("You must include at least one of these tokens: " + ", ".join(toks) + ".\n") if toks else ""

    narr_block = "\n".join(f"- {s}" for s in inside)
    template_rule = get_template_rule(tpl)
    fs_block = fewshot_block_for_template(tpl)
    first_token_hint = ""
    tl = (tpl or "").lower()
    if "how many" in tl or "quantity" in tl:
        first_token_hint = "Your question MUST start with the exact words 'How many' (or 'How much').\n"
    elif "how did i" in tl:
        first_token_hint = "Your question MUST start with the exact word 'How'.\n"
    elif "why did i" in tl:
        first_token_hint = "Your question MUST start with the exact word 'Why'.\n"
    elif "who did i" in tl or "people:" in tl:
        first_token_hint = "Your question MUST start with the exact word 'Who'.\n"
    elif "where did i put x" in tl or "where is object x" in tl or "in what location" in tl or "place:" in tl or "before / after" in tl:
        first_token_hint = "Your question MUST start with the exact word 'Where'.\n"
    elif "what" in tl: # Copre tutti i vari "What"
        first_token_hint = "Your question MUST start with the exact word 'What'.\n"

    prompt = (
        "You are a video QA assistant. Write exactly ONE short, natural English question.\n\n"
        + "=" * 70 + "\n"
        + f"TEMPLATE: {tpl}\n"
        + "=" * 70 + "\n"
        + f"{template_rule}"
        + first_token_hint
        + "=" * 70 + "\n"
        "GENERAL RULES:\n"
        "1. Write ONLY ONE question (3–18 words). End with a single question mark.\n"
        "2. Use AS MANY details from the narrations AS POSSIBLE.\n"
        "3. Do NOT include answers, hints, or prefixes like 'Question:' or 'Q:'.\n"
        "4. Do NOT write meta-text or refusals. Always produce a valid question.\n"
        "5. Use ONLY information from the narrations strictly inside the time window.\n"
        "6. Start your question with the required first word for this TEMPLATE, otherwise it will be rejected.\n"
        + anchor +
        "\nFEW-SHOTS (format guidance):\n"
        f"{fs_block}\n\n"
        "NARRATIONS IN THE WINDOW:\n"
        f"Time range: [{qs:.2f}s to {qe:.2f}s]\n"
        f"{narr_block}\n\n"
        "Now write the question:\n"
    )
    return prompt

# ---------- GBNF builders ----------
def _gbnf_common_chars():
    return r"[A-Za-z0-9 ,.'-]"

def _gbnf_question(prefix: str) -> str:
    chars = _gbnf_common_chars()
    return (
        f'root ::= q\n'
        f'q ::= "{prefix}" SP content "?"\n'
        f'content ::= {chars}{{1,120}}\n'
        f'SP ::= " "\n'
    )

def gbnf_where():
    return _gbnf_question("Where")

def gbnf_who():
    return _gbnf_question("Who")

def gbnf_howmany():
    chars = _gbnf_common_chars()
    return (
        'root ::= q\n'
        'q ::= "How" SP "many" SP content "?"\n'
        f'content ::= {chars}{{1,120}}\n'
        'SP ::= " "\n'
    )

def gbnf_what():
    return _gbnf_question("What")

# MODIFICA 4: Aggiunte GBNF per How e Why
def gbnf_how():
    # Semplice: inizia solo con "How". La validazione Python bloccherà "How many"
    return _gbnf_question("How")

def gbnf_why():
    return _gbnf_question("Why")


def gbnf_put_in():
    # GBNF "Put In" corretta
    content_chars = _gbnf_common_chars()
    content_str = f'({content_chars}*)' # 0 o più caratteri
    put_verb = '("put" | "place" | "drop" | "insert" | "pour" | "throw")'
    prep_in = '("in" | "into" | "inside" | "onto")'
    return (
        f'root ::= q\n'
        f'q ::= "What" SP {content_str} SP {put_verb} SP {content_str} SP {prep_in} SP {content_str} "?"\n'
        f'SP ::= " "\n'
    )


def gbnf_where_before_after():
    # GBNF "Before/After" corretta
    content_chars = _gbnf_common_chars()
    content_str = f'({content_chars}*)' # 0 o più caratteri
    return (
        f'root ::= q\n'
        f'q ::= "Where" SP {content_str} SP ("before" | "after") SP {content_str} "?"\n'
        f'SP ::= " "\n'
    )

# MODIFICA 5: Aggiornata `grammar_for_template` per coprire tutti i template
def grammar_for_template(tpl: str) -> str:
    if not USE_GRAMMAR:
        return ""
    tl = (tpl or "").lower()
    
    # Template specifici e problematici
    if "before / after" in tl:
        return gbnf_where_before_after()
    if "what did i put in x" in tl:
        return gbnf_put_in()
        
    # Template con parola chiave
    if "how many" in tl or "quantity" in tl:
        return gbnf_howmany()
    if "how did i" in tl:
        return gbnf_how()
    if "why did i" in tl:
        return gbnf_why()
    if "who did i" in tl or "people:" in tl:
        return gbnf_who()
    
    # Template "Where" generici
    if "where" in tl or "place:" in tl or "in what location" in tl:
        return gbnf_where()
        
    # Template "What" generici
    if "what" in tl or "state of" in tl:
        return gbnf_what()

    return "" # Default: nessuna grammatica

# ---------- ENHANCED VALIDATION (semantica leggera, no Unsure) ----------
# MODIFICA 6: Aggiornati _rx_starts e validate_query...
_rx_starts = {
    'where': re.compile(r'^\s*where\b', re.I),
    'who':   re.compile(r'^\s*who\b', re.I),
    'what':  re.compile(r'^\s*what\b', re.I),
    'how_many': re.compile(r'^\s*(how many|how much)\b', re.I),
    'how': re.compile(r'^\s*how\b', re.I), # Aggiunto
    'why': re.compile(r'^\s*why\b', re.I), # Aggiunto
}
def _starts_with(kind: str, q_lower: str) -> bool:
    return bool(_rx_starts[kind].search(q_lower))

def validate_query_for_template(template_name: str, query: str) -> Tuple[bool, str]:
    q = (query or "").strip(); ql = q.lower()
    if not q.endswith("?"): return False, "no_qmark"
    words = re.findall(r"\b\w+\b", q)
    if len(words) < 3 or len(words) > 24: return False, "bad_word_count"
    tl = (template_name or "").lower()

    # Controlli specifici per template
    if "before / after" in tl:
        if not _starts_with('where', ql): return False, "must_start_where"
        if not ("before" in ql or "after" in ql): return False, "missing_before_after"

    if "what did i put in x" in tl:
        if not _starts_with('what', ql): return False, "must_start_what"
        if not re.search(r"\b(put|place|drop|insert|pour|throw)\b", ql): return False, "missing_put_verb"
        if not re.search(r"\b(in|into|inside|onto)\b", ql): return False, "missing_container_prep"

    if "how many" in tl or "quantity" in tl:
        if not _starts_with('how_many', ql): return False, "must_start_how_many"

    if "how did i" in tl:
        if not _starts_with('how', ql): return False, "must_start_how"
        if _starts_with('how_many', ql): return False, "is_how_many_not_how" # Blocca "How many"

    if "where" in tl or "place:" in tl or "in what location" in tl:
        if not _starts_with('where', ql): return False, "must_start_where"

    if "who" in tl or "people:" in tl:
        if not _starts_with('who', ql): return False, "must_start_who"

    if "why did i" in tl:
        if not _starts_with('why', ql): return False, "must_start_why"

    if "what" in tl or "state of" in tl:
        if not _starts_with('what', ql): return False, "must_start_what"

    return True, "ok"

# ---------- VALIDAZIONE LIGHT + overlap (SENZA sinonimi, k=1) ----------
def is_valid_format(q: str, inside: List[str], tpl: str = "") -> Tuple[bool, str]:
    qn = (q or "").strip()
    if not qn: return False, "empty_text"
    if NO_QUESTION_RE.match(qn): return False, "llm_no_content"
    nwords = len(re.findall(r"\b\w+\b", qn))
    if nwords < 3 or nwords > 24: return False, "bad_len"
    if not qn.endswith("?"): return False, "no_qmark"
    if not inside: return False, "empty_inside"
    
    # MODIFICA 7: Overlap k=1 per TUTTI i template
    k = 1 
    
    if not has_min_overlap(qn, inside, k=k): return False, "low_overlap_inside"
    return True, "ok"

# ---------- Selezione con rerank ----------
FORBID_PUT = {"pick","take","open"}
def score_candidate(q: str, tpl: str, inside: List[str]) -> float:
    ok_tpl, _ = validate_query_for_template(tpl, q)
    ov = len(set(_tokens(q)) & set(_tokens(" ".join(inside)))) # Usa _tokens
    s = 1.0 if ok_tpl else 0.0
    s += 0.4*min(ov, 2)
    wc = len(re.findall(r"\b\w+\b", q))
    if 5 <= wc <= 14: s += 0.2
    tl = (tpl or "").lower()
    ql = (q or "").lower()
    if "what did i put in x" in tl:
        if any(v in ql for v in FORBID_PUT): s -= 0.6
        if not re.search(r"\b(in|into|inside|onto)\b", ql): s -= 0.6
    return s

def _infer_one(prompt: str, tpl: str, inside: List[str]) -> str:
    gram = grammar_for_template(tpl)
    trials = [
        dict(seed=42,  temperature=TEMPERATURE),
        dict(seed=123, temperature=(TEMP_BURST if USE_TEMP_BURST else max(0.1, TEMPERATURE))),
        dict(seed=777, temperature=(TEMP_BURST if USE_TEMP_BURST else max(0.2, TEMPERATURE))),
        dict(seed=314, temperature=(TEMP_BURST if USE_TEMP_BURST else max(0.3, TEMPERATURE))),
        dict(seed=271, temperature=(TEMP_BURST if USE_TEMP_BURST else max(0.25, TEMPERATURE))),
    ][:N_TRIALS]

    cands = []
    last_q = ""
    for tr in trials:
        try:
            txt = _chat_completion(prompt, max_tokens=MAX_NEW_TOKENS,
                                   seed=tr["seed"], temperature=tr["temperature"],
                                   grammar=gram)
            q = normalize_question(txt)
        except Exception as e:
            q = normalize_question(str(e))
        last_q = q
        fmt_ok, _ = is_valid_format(q, inside, tpl)
        if fmt_ok:
            cands.append(q)

    if not cands:
        return last_q
    return max(cands, key=lambda x: score_candidate(x, tpl, inside))

def generate_questions(prompts_with_tpl_and_inside: List[Tuple[str, str, List[str]]]) -> List[str]:
    out = [None] * len(prompts_with_tpl_and_inside)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futs = {ex.submit(_infer_one, p, t, ins): i for i, (p, t, ins) in enumerate(prompts_with_tpl_and_inside)}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
    return out

# ---------- MAIN (shard loop) ----------
shards = sorted([p for p in os.listdir(SHARDS_DIR) if p.endswith(".jsonl")])
print(f"[input] trovati {len(shards)} shard da processare.")
if not shards:
    raise SystemExit("Nessuno shard trovato.")

for si, shard_name in enumerate(shards, 1):
    in_path  = os.path.join(SHARDS_DIR, shard_name)
    out_path = os.path.join(OUT_DIR, f"queries_{shard_name}")

    seen_input_keys, seen_triplets = load_resume_sets(out_path)
    remaining_est = sum(1 for rec in read_jsonl(in_path) if make_input_key(rec) not in seen_input_keys)

    print(f"\n[shard {si}/{len(shards)}] input : {in_path}")
    print(f"                     output: {out_path}")
    print(f"   già fatti (resume): {len(seen_input_keys)}")
    print(f"   da processare:      {remaining_est}")

    if remaining_est == 0:
        print("   ✅ shard già completo. Skip.")
        csv_path = os.path.join(AUDIT_CSV_DIR, f"audit_{os.path.splitext(shard_name)[0]}.csv")
        if not os.path.exists(csv_path) and os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f_in, open(csv_path, "w", newline="", encoding="utf-8") as f_out:
                    w = csv.writer(f_out)
                    w.writerow(["video_uid","template","query","qs","qe","clip_s","clip_e"])
                    for line in f_in:
                        row = json.loads(line)
                        w.writerow([row.get("video_uid"), row.get("template"), row.get("query"),
                                    row.get("query_start_sec"), row.get("query_end_sec"),
                                    row.get("clip_start_sec"), row.get("clip_end_sec")])
                print(f"[audit] scritto: {csv_path}")
            except Exception as e:
                print(f"[audit] skip ({e})")
        continue

    pbar = tqdm(total=(remaining_est + BATCH_SIZE - 1)//BATCH_SIZE, desc="Batch nel shard")

    batch_items, batch_inside = [], []
    wrote = 0
    drop_counts = Counter()
    retry_counts = Counter()

    for rec in read_jsonl(in_path):
        key_in = make_input_key(rec)
        if key_in in seen_input_keys:
            continue

        inside = window_lines(rec, delta=DELTA_T, max_lines=MAX_INSIDE_LINES)
        if not inside:
            drop_counts["empty_inside"] += 1
            seen_input_keys.add(key_in)
            continue

        batch_items.append(rec)
        batch_inside.append(inside)

        if len(batch_items) < BATCH_SIZE:
            continue

        prompts = [
            (build_prompt_for_query(itm["matched_template"], float(itm["query_start_sec"]),
                                    float(itm["query_end_sec"]), ins), itm["matched_template"], ins)
            for itm, ins in zip(batch_items, batch_inside)
        ]
        qs_out = generate_questions(prompts)

        rows = []
        for itm, ins, q in zip(batch_items, batch_inside, qs_out):
            ok_fmt, why_fmt = is_valid_format(q, ins, itm["matched_template"])
            if not ok_fmt:
                drop_counts[f"fmt_{why_fmt}"] += 1
                seen_input_keys.add(make_input_key(itm))
                continue

            ok_sem, why_sem = validate_query_for_template(itm["matched_template"], q)
            if not ok_sem:
                drop_counts[f"sem_{why_sem}"] += 1
                retry_counts["semantic_retry_failed"] += 1
                seen_input_keys.add(make_input_key(itm))
                continue

            trip = (itm["video_uid"], itm["matched_template"], q.strip())
            if trip in seen_triplets:
                drop_counts["dup_text"] += 1
                seen_input_keys.add(make_input_key(itm))
                continue

            seen_triplets.add(trip)
            seen_input_keys.add(make_input_key(itm))
            rows.append({
                "video_uid": itm["video_uid"],
                "template": itm["matched_template"],
                "query": q.strip(),
                "clip_start_sec": float(itm.get("clip_start_sec", 0.0)),
                "clip_end_sec": float(itm.get("clip_end_sec", 0.0)),
                "query_start_sec": float(itm.get("query_start_sec", 0.0)),
                "query_end_sec": float(itm.get("query_end_sec", 0.0)),
                "source_narrations": ins
            })

        if rows:
            append_jsonl(out_path, rows, do_flush=True)
            wrote += len(rows)

        pbar.update(1)
        pbar.set_postfix(tot_write=wrote, ok_batch=len(rows), drop=sum(drop_counts.values()), retry_fail=retry_counts.get("semantic_retry_failed", 0))
        batch_items, batch_inside = [], []

    # Flush finale
    if batch_items:
        prompts = [
            (build_prompt_for_query(itm["matched_template"], float(itm["query_start_sec"]),
                                    float(itm["query_end_sec"]), ins), itm["matched_template"], ins)
            for itm, ins in zip(batch_items, batch_inside)
        ]
        qs_out = generate_questions(prompts)
        rows = []
        for itm, ins, q in zip(batch_items, batch_inside, qs_out):
            ok_fmt, why_fmt = is_valid_format(q, ins, itm["matched_template"])
            if not ok_fmt:
                drop_counts[f"fmt_{why_fmt}"] += 1
                seen_input_keys.add(make_input_key(itm))
                continue
            ok_sem, why_sem = validate_query_for_template(itm["matched_template"], q)
            if not ok_sem:
                drop_counts[f"sem_{why_sem}"] += 1
                retry_counts["semantic_retry_failed"] += 1
                seen_input_keys.add(make_input_key(itm))
                continue
            trip = (itm["video_uid"], itm["matched_template"], q.strip())
            if trip in seen_triplets:
                drop_counts["dup_text"] += 1
                seen_input_keys.add(make_input_key(itm))
                continue
            seen_triplets.add(trip)
            seen_input_keys.add(make_input_key(itm))
            rows.append({
                "video_uid": itm["video_uid"],
                "template": itm["matched_template"],
                "query": q.strip(),
                "clip_start_sec": float(itm.get("clip_start_sec", 0.0)),
                "clip_end_sec": float(itm.get("clip_end_sec", 0.0)),
                "query_start_sec": float(itm.get("query_start_sec", 0.0)),
                "query_end_sec": float(itm.get("query_end_sec", 0.0)),
                "source_narrations": ins
            })
        if rows:
            append_jsonl(out_path, rows, do_flush=True)
            wrote += len(rows)

    pbar.close()

    print("\n[drop & retry report per shard]")
    all_reasons = {**drop_counts, **retry_counts}
    for k, v in sorted(all_reasons.items(), key=lambda x: -x[1])[:30]:
        print(f"  - {k}: {v}")

    # Audit CSV
    csv_path = os.path.join(AUDIT_CSV_DIR, f"audit_{os.path.splitext(shard_name)[0]}.csv")
    try:
        with open(out_path, "r", encoding="utf-8") as f_in, open(csv_path, "w", newline="", encoding="utf-8") as f_out:
            w = csv.writer(f_out)
            w.writerow(["video_uid","template","query","qs","qe","clip_s","clip_e"])
            for line in f_in:
                row = json.loads(line)
                w.writerow([row.get("video_uid"), row.get("template"), row.get("query"),
                            row.get("query_start_sec"), row.get("query_end_sec"),
                            row.get("clip_start_sec"), row.get("clip_end_sec")])
        print(f"[audit] scritto: {csv_path}")
    except Exception as e:
        print(f"[audit] skip ({e})")

print("\n--- SCRIPT CONCLUSO CORRETTAMENTE (v3.3) ---")
print(f"Output in: {OUT_DIR}")