# ===========================================
# SANITIZE + SHARD robusto (JSON / JSONL / .gz)
# - Ripara: NaN/Infinity, virgole finali, caratteri non-ascii
# - Supporta: root array, root oggetto con "clips"/"data", o JSONL grezzo
# - Output: *.clean.jsonl (+ shards )
# ===========================================

import os
import io
import re
import json
import math
import gzip
import glob
from typing import Any, Dict, List, Tuple
from collections import Counter


IN_PATH    = "pretrain_catalog/candidate/pretrain_catalog_v2_multi_flat.json"

# Dove salvare il file intermedio pulito
OUT_JSONL  = "shards/v2/annotations/pretrain_catalog_v2_multi_flat.clean.jsonl"

# Se dividere in shard per parallelismo
MAKE_SHARDS = True
SHARD_DIR   = "shards/v2/annotations/shards_flat"
SHARD_SIZE  = 10000  # Righe per shard

# Soglie di validazione
EPS       = 1e-6
MIN_Q_LEN = 0.2    # Durata minima query (secondi)
MAX_DUR   = 120.0  # Durata massima ragionevole per una query (secondi)

# ======= JSON BACKEND =======
# Usa 'orjson' se disponibile (molto più veloce), altrimenti standard 'json'
try:
    import orjson as _json
    def jloads_bytes(b: bytes): return _json.loads(b)
    def jdumps(obj): return _json.dumps(obj)
    JSON_IMPL = "orjson"
except ImportError:
    def jloads_bytes(b: bytes): return json.loads(b.decode("utf-8"))
    def jdumps(obj): return json.dumps(obj, ensure_ascii=False).encode("utf-8")
    JSON_IMPL = "stdlib-json"
print(f"[INFO] JSON backend in uso: {JSON_IMPL}")

# ======= HELPER I/O =======
def _open_read(path: str):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")

def _open_write(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return gzip.open(path, "wb") if path.endswith(".gz") else open(path, "wb")

# ======= FUNZIONI DI PULIZIA (SANITIZERS) =======
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def _finite_or_none(x: Any):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x

def _trim_str(s: Any):
    return s.strip() if isinstance(s, str) else s

def sanitize_value(v: Any):
    """Rimuove ricorsivamente None/NaN/Inf e pulisce stringhe."""
    if _is_number(v):
        return _finite_or_none(v)
    if isinstance(v, str):
        s = _trim_str(v)
        return s if s != "" else None
    if isinstance(v, list):
        out = []
        for el in v:
            sv = sanitize_value(el)
            if sv is not None:
                out.append(sv)
        return out
    if isinstance(v, dict):
        out = {}
        for k, val in v.items():
            if val is None:
                continue
            sv = sanitize_value(val)
            if sv is not None:
                out[k] = sv
        return out
    return None

def _num(x):
    return float(x) if _is_number(x) else None

def validate_flat(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """Controlla che il record abbia tutti i campi necessari e valori validi."""
    req = ["video_uid", "clip_start_sec", "clip_end_sec", "query_start_sec", "query_end_sec", "matched_template", "context_text"]
    missing = [k for k in req if k not in rec]
    if missing: 
        return False, f"missing_fields:{','.join(missing)}"

    cs, ce = _num(rec.get("clip_start_sec")), _num(rec.get("clip_end_sec"))
    qs, qe = _num(rec.get("query_start_sec")), _num(rec.get("query_end_sec"))
    
    if None in (cs, ce, qs, qe):
        return False, "timestamps_none"
    if not (0 <= cs < ce):
        return False, "clip_bounds_invalid"
    # Tolleranza EPS per floating point errors
    if not (cs - EPS <= qs < qe <= ce + EPS):
        return False, "query_out_of_clip_bounds"
    
    qlen = qe - qs
    if qlen < MIN_Q_LEN:
        return False, "query_too_short"
    if qlen > MAX_DUR:
        return False, "query_too_long"

    if not isinstance(rec.get("matched_template"), str) or not rec["matched_template"]:
        return False, "template_empty"
    if not isinstance(rec.get("video_uid"), str) or not rec["video_uid"]:
        return False, "video_uid_empty"
    
    # "narrations" è opzionale ma se c'è deve essere lista
    if "narrations" in rec and not isinstance(rec["narrations"], list):
        return False, "narrations_not_list"
        
    return True, "ok"

# ======= RIPARAZIONE JSON TEXT =======
_re_nan_inf = re.compile(r'(?<!")\b(NaN|Infinity|-Infinity)\b')
_re_trailing_commas = re.compile(r',(\s*[\]\}])')
_re_ctrl = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]')

def repair_json_text(txt: str) -> str:
    """Tenta di riparare errori comuni nei JSON corrotti."""
    # 1) Normalizza newline
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    # 2) Sostituisci NaN/Infinity con null (standard JSON)
    txt = _re_nan_inf.sub("null", txt)
    # 3) Rimuovi caratteri di controllo illegali
    txt = _re_ctrl.sub("", txt)
    # 4) Rimuovi virgole finali (es: [1, 2,] -> [1, 2])
    txt = _re_trailing_commas.sub(r"\1", txt)
    return txt

def _first_non_ws(b: bytes) -> int:
    for i, ch in enumerate(b):
        if chr(ch) not in " \t\r\n":
            return i
    return -1

# ======= LETTORE ROBUSTO (Iterator) =======
def iter_items(path: str):
    """Generatore che tenta di leggere record da qualsiasi formato JSON/JSONL."""
    if not os.path.exists(path):
        print(f"[WARN] File non trovato: {path}")
        return

    raw = None
    try:
        with _open_read(path) as f:
            raw = f.read()
    except Exception as e:
        print(f"[ERR] Errore lettura file {path}: {e}")
        return

    if not raw:
        return

    # Heuristica: JSON o JSONL?
    start = _first_non_ws(raw)
    looks_json = start >= 0 and raw[start:start+1] in (b"[", b"{")

    # --- STRATEGIA 1: Parsing come JSON completo ---
    if looks_json:
        try:
            obj = jloads_bytes(raw)
            # Caso A: Lista di record [ {...}, {...} ]
            if isinstance(obj, list):
                for it in obj: yield it
                return
            # Caso B: Oggetto contenitore { "clips": [...] }
            if isinstance(obj, dict):
                # Cerca chiavi comuni per liste di dati
                for key in ("clips", "data", "rows", "items", "candidates"):
                    if key in obj and isinstance(obj[key], list):
                        for it in obj[key]: yield it
                        return
                # Caso C: Oggetto singolo
                yield obj
                return
        except Exception:
            # Se fallisce il parsing standard, tenta la riparazione del testo
            pass

        # --- STRATEGIA 2: Riparazione testo e riprova ---
        try:
            txt = raw.decode("utf-8", errors="ignore")
            fixed = repair_json_text(txt)
            obj = json.loads(fixed)
            if isinstance(obj, list):
                for it in obj: yield it
                return
            if isinstance(obj, dict):
                for key in ("clips", "data", "rows", "items", "candidates"):
                    if key in obj and isinstance(obj[key], list):
                        for it in obj[key]: yield it
                        return
                yield obj
                return
        except Exception:
            pass 

    # --- STRATEGIA 3: Parsing riga per riga (JSONL) ---
    # Fallback finale: prova a trattarlo come JSON Lines, anche se sembrava un JSON rotto
    stream = io.BytesIO(raw)
    for line in stream:
        line = line.strip()
        if not line: continue
        try:
            yield jloads_bytes(line)
        except Exception:
            try:
                # Ultimo tentativo: ripara la singola riga
                fixed = repair_json_text(line.decode("utf-8", errors="ignore"))
                yield json.loads(fixed)
            except Exception:
                continue # Riga irrecuperabile

# ======= PROCESSO PRINCIPALE =======
def sanitize_to_jsonl(in_path: str, out_jsonl: str) -> Dict[str,int]:
    kept = 0
    dropped = 0
    total = 0
    reasons = Counter()
    dropped_fields = Counter()

    print(f"--- Inizio Sanitize ---")
    print(f"Input: {in_path}")
    
    with _open_write(out_jsonl) as w:
        for rec in iter_items(in_path):
            total += 1
            
            # Statistica campi nulli prima della pulizia
            if isinstance(rec, dict):
                for k, v in rec.items():
                    if v is None or (isinstance(v, float) and not math.isfinite(v)):
                        dropped_fields[k] += 1

            # 1. Pulisci valori
            clean = sanitize_value(rec)
            if not isinstance(clean, dict):
                dropped += 1
                reasons["not_dict_after_sanitize"] += 1
                continue

            # 2. Valida struttura
            ok, why = validate_flat(clean)
            if not ok:
                dropped += 1
                reasons[why] += 1
                continue

            # 3. Scrivi
            w.write(jdumps(clean))
            w.write(b"\n")
            kept += 1

    print(f"\n=== REPORT ===")
    print(f"Totale record letti: {total}")
    print(f" Record validi:     {kept}")
    print(f" Record scartati:   {dropped}")
    
    if reasons:
        print("\nMotivi scarto:")
        for r, c in reasons.most_common():
            print(f" - {r}: {c}")
            
    return {"total": total, "kept": kept, "dropped": dropped}

def shard_jsonl(in_jsonl: str, shard_dir: str, shard_size: int) -> List[str]:
    """Divide un file JSONL in più file più piccoli."""
    os.makedirs(shard_dir, exist_ok=True)
    
    # Pulisci vecchi shard
    for old in glob.glob(os.path.join(shard_dir, "shard_*.jsonl")):
        try: os.remove(old)
        except: pass

    shard_idx = 0
    count = 0
    paths = []
    out = None

    def _open_shard(idx):
        path = os.path.join(shard_dir, f"shard_{idx:04d}.jsonl")
        return path, open(path, "wb")

    print(f"\n--- Creazione Shards ---")
    with _open_read(in_jsonl) as f:
        for line in f:
            if out is None or (count % shard_size == 0):
                if out is not None: out.close()
                path, out = _open_shard(shard_idx)
                paths.append(path)
                shard_idx += 1
            out.write(line)
            count += 1
            
    if out is not None: 
        out.close()

    print(f"Creati {len(paths)} shard in: {shard_dir}")
    print(f"Dimensione media: {shard_size} righe")
    return paths


if __name__ == "__main__":
    # Esegui sanitizzazione
    stats = sanitize_to_jsonl(IN_PATH, OUT_JSONL)
    
    # Esegui sharding se ci sono dati validi
    if MAKE_SHARDS and stats["kept"] > 0:
        shard_paths = shard_jsonl(OUT_JSONL, SHARD_DIR, SHARD_SIZE)
    else:
        shard_paths = [OUT_JSONL]

    print("\n[COMPLETATO] Pipeline terminata con successo.")