# ----------------------------------------------------------------------
# SCRIPT 2: generate_clips_achnor.py (TRQ-ANCHOR + TRQ-FILL su evento DB)
#
# Scopo: Esegue l'allocazione multiquery (FASE 4, 5, Report).
#        - Anchor: campionata dal TRQ (template, d, qs_rel, qe_rel, clen_train)
#        - Match anchor: evento nel DB con stesso template, durata >= d, densityscore alto
#        - Clip: costruita attorno all'anchor usando d_rel e posizioni relative
#        - Fill: altre query da TRQ di clip di lunghezza simile, piazzate nella stessa clip
#
# Uso (Debug Veloce):
# > python generate_clips_anchor.py --geom_only
#
# Uso (Produzione):
# > python generate_clips_anchor.py
# ----------------------------------------------------------------------

import os, json, math, random, argparse, sqlite3
from scipy.stats import ks_2samp
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
BASE_PATH = ""
TRAIN_JSON_PATH = os.path.join(BASE_PATH, "nlq_train.json")
NARR_JSON_PATH  = os.path.join(BASE_PATH, "narration.json")
CATALOG_PATH    = "pretrain_catalog/"
CATALOG_DB_PATH = os.path.join(CATALOG_PATH, "catalog.db")
OUT_MULTI_PATH  = os.path.join(CATALOG_PATH, "candidate/pretrain_catalog_v2_multi.json")
OUT_FLAT_PATH   = os.path.join(CATALOG_PATH, "candidate/pretrain_catalog_v2_multi_flat.json")

FINAL_TARGET = 150000
SAFETY       = 1.10
ACC_PRIOR    = np.array([1,1,1,1,1,0.9], dtype=float)

MIN_QUERY_LEN_S = 0.4
DURATION_BUCKETS = [0,2,4,8,16,32,60]
NARR_EPS = 1e-3
DEBUG_ALLOC = True 

# --- COSTANTI  ---
CLEN_TOLERANCE_S = 10.0  
ALIGNMENT_TOLERANCE_S = 12.0
HOLISTIC_MAX_Q_PER_CLIP = 10      
HOLISTIC_MAX_CLIP_ATTEMPTS_FACTOR = 0.3 
HOLISTIC_MIN_CLIP_ATTEMPTS = 1000
DB_CANDIDATE_SEARCH_LIMIT = 1000000
ANCHOR_MAX_SAMPLE = 100000          # numero massimo eventi DB da considerare per anchor
FILL_DURATION_TOL_FRAC = 0.15       # tolleranza relativa su durata fill (±15%)

print(f"--- Inizializzazione Allocazione Olistica ---")
print(f"Lettura DB da: {CATALOG_DB_PATH}")

# --- Parsing Argomenti ---
parser = argparse.ArgumentParser(description="Script per l'allocazione Olistica da catalog.db.")
parser.add_argument(
    "--geom_only",
    action="store_true",
    help="Modalità Debug Veloce: Usa query SQL --geom_only (ignora i template)."
)
args = parser.parse_args()
GEOM_ONLY_MODE = args.geom_only

if GEOM_ONLY_MODE:
    print("[ATTENZIONE] Esecuzione in modalità --geom_only (Debug Veloce).")
else:
    print("[INFO] Esecuzione in modalità Produzione (Completa).")

# Seed
rng = np.random.default_rng(1234)
random.seed(1234)

# ---------- UTILS ----------
def canonical_bucket(b):
    if isinstance(b, (tuple, list, np.ndarray)):
        b = (float(b[0]), float(b[1]))
        return (b[0], b[1])
    a, c = b
    return (float(a), float(c))

def bucket_of(d, edges=DURATION_BUCKETS):
    for i in range(len(edges)-1):
        if edges[i] <= d < edges[i+1]:
            return canonical_bucket((edges[i], edges[i+1]))
    return canonical_bucket((edges[-2], edges[-1]))

def _dedup_segments(pairs, tol=0.05):
    if not pairs: return []
    pairs = [(float(s), float(e)) for (s,e) in pairs if (s is not None and e is not None and float(e) > float(s))]
    pairs.sort()
    out = []
    for s,e in pairs:
        if not out: 
            out.append([s,e]); 
            continue
        ps,pe = out[-1]
        if abs(s-ps) <= tol and abs(e-pe) <= tol:
            out[-1][0] = 0.5*(ps+s); out[-1][1] = 0.5*(pe+e)
        else:
            out.append([s,e])
    return [(s,e) for s,e in out]

def get_all_narrations_by_video(narr_json_path):
    if not os.path.exists(narr_json_path):
        raise FileNotFoundError(narr_json_path)
    with open(narr_json_path, 'r') as f:
        return json.load(f)

# ---------- FASE 1/5: Caricamento TRQ e G_Plan ----------

print("[FASE 1/5] Caricamento training set & narrazioni...")
def get_target_distributions(train_json_path):
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(train_json_path)
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    templ, recs = [], []
    for v in train_data['videos']:
        for c in v.get('clips', []):
            cs = c.get("clip_start_sec", None); ce = c.get("clip_end_sec", None)
            if cs is None or ce is None: continue
            csf, cef = float(cs), float(ce); Lc = float(cef - csf)
            if Lc <= 0: continue
            for a in c.get('annotations', []):
                for q in a.get('language_queries', []):
                    s = q.get("clip_start_sec", None); e = q.get("clip_end_sec",   None)
                    if s is None or e is None:
                        sv = q.get("video_start_sec", None); ev = q.get("video_end_sec",   None)
                        cvs = c.get("video_start_sec", None)
                        if sv is None or ev is None or cvs is None: continue
                        s = (float(sv) - float(cvs)) + csf; e = (float(ev) - float(cvs)) + csf
                    s = float(s); e = float(e); Lq = e - s
                    if Lq <= 0: continue
                    qs_rel = (s - csf) / Lc; qe_rel = (e - csf) / Lc
                    if not (0.0 <= qs_rel <= 1.0 and 0.0 <= qe_rel <= 1.0): continue
                    d_rel = qe_rel - qs_rel
                    if not (0.0 < d_rel <= 1.0): continue
                    templ.append(q.get('template'))
                    recs.append(Lq)
    template_dist = pd.Series([t for t in templ if t is not None]).value_counts(normalize=True).to_dict()
    original_durations = np.array(recs, dtype=np.float32)
    return template_dist, original_durations, train_data

def build_TRQ_from_train(train_obj, rel_dedup_tol=0.05):
    """
    TRQ: una riga per ogni query del train:
    - video_uid, clip_cs, clip_ce, clen
    - qs, qe, d
    - qs_rel, qe_rel
    - bucket (per d)
    - template (per matching con G_plan)
    """
    rows = []
    for v in train_obj["videos"]:
        vid = v.get("video_uid")
        for c in v.get("clips", []):
            cs = float(c["clip_start_sec"]); ce = float(c["clip_end_sec"])
            clen = max(1e-9, ce - cs)
            all_pairs = []
            for a in c.get("annotations", []):
                for q in a.get("language_queries", []):
                    tpl = q.get("template", None)
                    s = q.get("clip_start_sec", None); e = q.get("clip_end_sec",   None)
                    if s is None or e is None:
                        sv = q.get("video_start_sec", None); ev = q.get("video_end_sec",   None)
                        cvs = c.get("video_start_sec", None)
                        if sv is None or ev is None or cvs is None: continue
                        s = (float(sv) - float(cvs)) + cs; e = (float(ev) - float(cvs)) + cs
                    else:
                        s, e = float(s), float(e)
                    if cs <= s < e <= ce:
                        all_pairs.append((s, e, tpl))
            if not all_pairs:
                continue
            # dedup per geometria (solo s,e)
            segs = [(s, e) for (s, e, _) in all_pairs]
            uniq = _dedup_segments(segs, tol=rel_dedup_tol)
            # per semplicità, assegniamo il template originale del primo match di (s,e)
            tpl_map = {}
            for (s, e, tpl) in all_pairs:
                tpl_map.setdefault((round(s,3), round(e,3)), tpl)
            for (s, e) in uniq:
                d = e - s
                qs_rel = (s - cs) / clen
                qe_rel = (e - cs) / clen
                tpl = tpl_map.get((round(s,3), round(e,3)), None)
                rows.append({
                    "video_uid": vid, "clip_cs": cs, "clip_ce": ce, "clen": clen,
                    "qs": s, "qe": e, "d": d,
                    "qs_rel": qs_rel, "qe_rel": qe_rel,
                    "template": tpl
                })
    TRQ = pd.DataFrame(rows)
    mask_valid = TRQ["d"].notna()
    TRQ.loc[mask_valid, "bucket"] = TRQ.loc[mask_valid, "d"].astype(float).apply(lambda x: bucket_of(float(x)))
    return TRQ

template_distribution, original_durations, train_json = \
    get_target_distributions(TRAIN_JSON_PATH)
TRQ = build_TRQ_from_train(train_json)
print(f"TRQ costruito: {len(TRQ)} righe, {TRQ['d'].notna().sum()} query valide.")

# --- Pre-calcolo Durata Video ---
print("Pre-calcolo durata stimata video (basata su ultima narrazione)...")
all_narrations = get_all_narrations_by_video(NARR_JSON_PATH)
video_max_ts = {}
VIDEO_END_BUFFER_S = 5.0 
for vid, data in tqdm(all_narrations.items(), desc="Calcolo max timestamp"):
    narrs = data.get("narration_pass_1", {}).get("narrations", [])
    max_time = 0.0
    if narrs:
        try:
            valid_ts = [float(n['timestamp_sec']) for n in narrs if n.get('timestamp_sec') is not None]
            if valid_ts: max_time = max(valid_ts)
        except (ValueError, TypeError): pass 
    video_max_ts[vid] = max_time + VIDEO_END_BUFFER_S 
print(f"Durate stimate calcolate per {len(video_max_ts)} video.")
del all_narrations

# ---------- FASE 2/5: Costruzione G_Plan ----------

print("\n[FASE 2/5] Costruzione target congiunto & piano...")
def build_joint_budget(train_obj, total_queries, buckets, geom_only=False):
    counts = defaultdict(int)
    for v in train_obj["videos"]:
        for c in v["clips"]:
            for a in c["annotations"]:
                for q in a.get("language_queries", []):
                    t = q.get('template')
                    s = q.get("video_start_sec", q.get("clip_start_sec"))
                    e = q.get("video_end_sec",   q.get("clip_end_sec"))
                    if s is None or e is None: continue
                    if not geom_only and t is None: continue 
                    d = float(e) - float(s)
                    if d <= 0: continue
                    b = bucket_of(d, edges=buckets)
                    if geom_only:
                        key = b 
                    else:
                        key = (t, b) 
                    counts[key] += 1
                    
    keys = list(counts.keys())
    vals = np.array([counts[k] for k in keys], dtype=float)
    probs = vals / max(1, vals.sum())
    scaled = probs * total_queries
    floors = np.floor(scaled).astype(int)
    r = total_queries - floors.sum()
    order = np.argsort(-(scaled - floors))
    for i in range(r):
        floors[order[i]] += 1
    return {keys[i]: int(floors[i]) for i in range(len(keys))}

def plan_joint_generation(T_joint, acc_prior_bucket, safety, geom_only=False):
    edges = [canonical_bucket((DURATION_BUCKETS[i], DURATION_BUCKETS[i+1]))
             for i in range(len(DURATION_BUCKETS)-1)]
    idx_of_bucket = {edges[i]: i for i in range(len(edges))}
    G = {}
    for key, tgt in T_joint.items():
        if geom_only:
            b = key
        else:
            t, b = key
        pacc = float(acc_prior_bucket[idx_of_bucket[b]])
        pacc = max(1e-3, min(1.0, pacc))
        G[key] = int(math.ceil(tgt / pacc * safety))
    return G

T_joint = build_joint_budget(train_json, FINAL_TARGET, DURATION_BUCKETS, geom_only=GEOM_ONLY_MODE)
G_plan  = plan_joint_generation(T_joint, ACC_PRIOR, SAFETY, geom_only=GEOM_ONLY_MODE)
print(f"T_joint (celle): {len(T_joint)} | target totale={sum(T_joint.values())}")
print(f"G_plan  (celle): {len(G_plan)}  | grezzo con safety={sum(G_plan.values())}")

# ---------- FASE 3/5: Preparazione DB ----------

print("\n[FASE 3/5] Connessione al DB e preparazione...")
def connect_to_db(db_path):
    if not os.path.exists(db_path):
        print(f"ERRORE: Database '{db_path}' non trovato.")
        print("Esegui prima 'create_catalog_db.py'.")
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = 0")
        conn.execute("PRAGMA cache_size = -1000000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.row_factory = sqlite3.Row 
        return conn
    except Exception as e:
        print(f"ERRORE: Impossibile connettersi al DB: {e}")
        return None

def reset_db_usage(db_conn):
    try:
        print("Reset dello stato 'usato' nel DB...")
        cursor = db_conn.cursor()
        cursor.execute("UPDATE candidates SET usato = 0 WHERE usato = 1")
        db_conn.commit()
        print(f"Reset completato: {cursor.rowcount} righe aggiornate.")
    except Exception as e:
        print(f"ERRORE durante il reset del DB: {e}")

db_conn = connect_to_db(CATALOG_DB_PATH)
if db_conn is None:
    raise SystemExit(1)

reset_db_usage(db_conn)

# ---------- HELPER ALLOCAZIONE ----------

def _overlap_abs(iv1, iv2, tol=1e-3):
    return not (iv1[1] <= iv2[0] + tol or iv1[0] >= iv2[1] - tol)

def mark_candidates_used(db_conn, used_id_set):
    if not used_id_set: 
        return
    try:
        cursor = db_conn.cursor()
        id_tuples = [(int(id_),) for id_ in used_id_set]
        cursor.executemany("UPDATE candidates SET usato = 1 WHERE id = ?", id_tuples)
        db_conn.commit()
    except Exception as e:
        print(f"ERRORE durante l'aggiornamento 'usato' nel DB: {e}")

def get_next_target(g_plan_remaining, geom_only):
    try:
        if not g_plan_remaining: 
            return None, None
        target_k = max((k for k, v in g_plan_remaining.items() if v > 0), key=g_plan_remaining.get)
        if geom_only: 
            return target_k, None 
        else: 
            return target_k[1], target_k[0] 
    except ValueError: 
        return None, None 

def sample_anchor_from_TRQ(TRQ_full, target_bucket, target_template, geom_only):
    """
    Campiona una riga dal TRQ compatibile con (template,bucket).
    """
    df = TRQ_full[TRQ_full["bucket"] == target_bucket].copy()
    if not geom_only:
        df = df[df["template"] == target_template]
    if df.empty:
        return None
    idx = rng.integers(0, len(df))
    return df.iloc[idx]

# ---------- FASE 4: Allocazione  (TRQ Anchor + TRQ Fill) ----------
print("\n[FASE 4/5] Avvio allocazione multiquery (TRQ-ANCHOR + TRQ-FILL)...")
def allocate_holistic_trq_anchor_fill(
    db_conn, G_plan, TRQ_full, video_max_ts_lookup, geom_only
):
    G_plan_remaining = G_plan.copy()
    placed_by_cell = defaultdict(int)
    placed_by_bucket = defaultdict(int)
    clips_multi = [] 
    
    total_targets_remaining = sum(G_plan_remaining.values())
    max_clip_creation_attempts = max(
        HOLISTIC_MIN_CLIP_ATTEMPTS, 
        int(total_targets_remaining * HOLISTIC_MAX_CLIP_ATTEMPTS_FACTOR)
    )
    pbar = tqdm(total=total_targets_remaining, desc="Allocazione Olistica (Query)")
    clip_attempts = 0 
    
    while total_targets_remaining > 0 and clip_attempts < max_clip_creation_attempts:
        # Target corrente (bucket, template)
        target_bucket, target_template = get_next_target(G_plan_remaining, geom_only)
        if target_bucket is None:
            break

        # 1) Anchor dal TRQ (T, d...)
        anchor_TRQ = sample_anchor_from_TRQ(TRQ_full, target_bucket, target_template, geom_only)
        if anchor_TRQ is None:
            # impossibile soddisfare questa cella: azzera
            target_key = target_bucket if geom_only else (target_template, target_bucket)
            G_plan_remaining[target_key] = 0
            clip_attempts += 1
            continue

        # 2) Match anchor con evento DB (evento lungo >= d_train)
        d_train = float(anchor_TRQ["d"])
        qs_rel  = float(anchor_TRQ["qs_rel"])
        qe_rel  = float(anchor_TRQ["qe_rel"])
        d_rel   = qe_rel - qs_rel
        if d_rel <= 0:
            clip_attempts += 1
            continue

        bucket_anchor = canonical_bucket(anchor_TRQ["bucket"])
        tpl_anchor    = anchor_TRQ.get("template", None)

        cursor = db_conn.cursor()
        if geom_only:
            sql_anchor = (
                "SELECT id, video_uid, cs, ce, _len, narrations_json, density_score "
                "FROM candidates "
                "WHERE bucket_lo = ? AND bucket_hi = ? "
                "  AND _len >= ? AND usato = 0 "
                "ORDER BY density_score DESC LIMIT ?"
            )
            params_anchor = [bucket_anchor[0], bucket_anchor[1], d_train, ANCHOR_MAX_SAMPLE]
        else:
            sql_anchor = (
                "SELECT c.id, c.video_uid, c.cs, c.ce, c._len, c.narrations_json, c.density_score "
                "FROM candidates c JOIN templates t ON c.id = t.candidate_id "
                "WHERE t.template_name = ? AND c.bucket_lo = ? AND c.bucket_hi = ? "
                "  AND c._len >= ? AND c.usato = 0 "
                "ORDER BY c.density_score DESC LIMIT ?"
            )
            params_anchor = [tpl_anchor, bucket_anchor[0], bucket_anchor[1], d_train, ANCHOR_MAX_SAMPLE]

        rows_anchor = cursor.execute(sql_anchor, params_anchor).fetchall()
        if not rows_anchor:
            clip_attempts += 1
            continue

        C_anchor  = random.choice(rows_anchor)
        anchor_id = C_anchor["id"]
        video_uid = C_anchor["video_uid"]
        cs_event  = float(C_anchor["cs"])
        ce_event  = float(C_anchor["ce"])
        len_event = float(C_anchor["_len"])
        narr_json = C_anchor["narrations_json"]

        # 2.a) Slice di query dentro l'evento: [cs_event, cs_event + d_train] (troncata se serve)
        qs_anchor = cs_event
        qe_anchor = min(qs_anchor + d_train, ce_event)
        d_final   = qe_anchor - qs_anchor
        if d_final < MIN_QUERY_LEN_S - NARR_EPS:
            clip_attempts += 1
            continue

        # 2.b) Costruisci la clip attorno alla query per rispettare qs_rel, qe_rel del TRQ
        #     qs_anchor = clip_cs + qs_rel * L_clip
        #     qe_anchor = clip_cs + qe_rel * L_clip
        #  => L_clip = d_final / d_rel, clip_cs = qs_anchor - qs_rel * L_clip
        L_clip  = max(d_final / max(d_rel, 1e-6), d_final)
        clip_cs = qs_anchor - qs_rel * L_clip
        clip_ce = clip_cs + L_clip

        # controllo consistenza (query dentro clip)
        if not (clip_cs - NARR_EPS <= qs_anchor < qe_anchor <= clip_ce + NARR_EPS):
            clip_attempts += 1
            continue

        # check durata video (clip dentro [0, video_max_ts])
        max_ts = video_max_ts_lookup.get(video_uid, 0.0)
        if clip_ce > max_ts + NARR_EPS:
            mark_candidates_used(db_conn, {anchor_id})
            clip_attempts += 1
            continue

        # Stato del clip corrente
        clip_occupied = [(qs_anchor, qe_anchor)]
        candidati_usati = {anchor_id}
        query_piazzate = []

        # Query anchor
        anchor_narrs = json.loads(narr_json)
        Q_anchor = {
            "video_uid": video_uid,
            "matched_template": (tpl_anchor if not geom_only else "GEOM_ONLY"),
            "narrations": anchor_narrs,
            "query_start_sec": float(qs_anchor),
            "query_end_sec": float(qe_anchor),
            "clip_start_sec": float(clip_cs),
            "clip_end_sec": float(clip_ce),
            "bucket": bucket_anchor,
        }
        query_piazzate.append(Q_anchor)

        target_key = bucket_anchor if geom_only else (tpl_anchor, bucket_anchor)
        if target_key in G_plan_remaining and G_plan_remaining[target_key] > 0:
            G_plan_remaining[target_key] -= 1
            if G_plan_remaining[target_key] == 0:
                del G_plan_remaining[target_key]
        placed_by_cell[target_key] += 1
        placed_by_bucket[bucket_anchor] += 1
        total_targets_remaining -= 1
        pbar.update(1)

        # 3) FILL: siblings TRQ + eventi reali DB
        TRQ_fill = TRQ_full[
            np.abs(TRQ_full["clen"] - L_clip) <= CLEN_TOLERANCE_S
        ].copy()

        if TRQ_fill.empty:
            # clip solo con anchor
            final_clip_obj = {
                "video_uid": video_uid,
                "clip_start_sec": clip_cs,
                "clip_end_sec": clip_ce,
                "narrations": [],
                "queries": query_piazzate,
            }
            clips_multi.append(final_clip_obj)
            mark_candidates_used(db_conn, candidati_usati)
            clip_attempts = 0
            continue

        TRQ_fill = TRQ_fill.sample(frac=1.0, random_state=rng.integers(0, 1e9))

        # 4) Loop di fill — ogni sibling deve corrispondere a un evento reale NELLO STESSO VIDEO
        for _, g in TRQ_fill.iterrows():
            if len(query_piazzate) >= HOLISTIC_MAX_Q_PER_CLIP:
                break

            tpl_i = g.get("template", None)
            bucket_i = canonical_bucket(g["bucket"])
            qs_rel_i = float(g["qs_rel"])
            qe_rel_i = float(g["qe_rel"])
            d_rel_i = qe_rel_i - qs_rel_i
            if d_rel_i <= 0:
                continue

            d_target = d_rel_i * L_clip
            if d_target < MIN_QUERY_LEN_S - NARR_EPS:
                continue

            d_orig = float(g["d"])
            if d_orig > 0:
                frac_diff = abs(d_target - d_orig) / d_orig
                if frac_diff > FILL_DURATION_TOL_FRAC:
                    continue

            qs_abs = clip_cs + qs_rel_i * L_clip
            qe_abs = qs_abs + d_target

            if not (clip_cs - NARR_EPS <= qs_abs < qe_abs <= clip_ce + NARR_EPS):
                continue

            if any(_overlap_abs((qs_abs, qe_abs), occ, tol=1e-3) for occ in clip_occupied):
                continue

            # 4.1) Evento DB per sibling (STESSO video dell'anchor)
            cursor = db_conn.cursor()
            if geom_only:
                sql_sib = (
                    "SELECT id, video_uid, cs, ce, _len, narrations_json, density_score "
                    "FROM candidates "
                    "WHERE video_uid = ? "
                    "  AND bucket_lo = ? AND bucket_hi = ? "
                    "  AND _len >= ? AND usato = 0 "
                    "ORDER BY density_score DESC LIMIT ?"
                )
                params_sib = [video_uid, bucket_i[0], bucket_i[1], d_target, DB_CANDIDATE_SEARCH_LIMIT]
            else:
                sql_sib = (
                    "SELECT c.id, c.video_uid, c.cs, c.ce, c._len, c.narrations_json, c.density_score "
                    "FROM candidates c JOIN templates t ON c.id = t.candidate_id "
                    "WHERE c.video_uid = ? "
                    "  AND t.template_name = ? AND c.bucket_lo = ? AND c.bucket_hi = ? "
                    "  AND c._len >= ? AND c.usato = 0 "
                    "ORDER BY c.density_score DESC LIMIT ?"
                )
                params_sib = [video_uid, tpl_i, bucket_i[0], bucket_i[1], d_target, DB_CANDIDATE_SEARCH_LIMIT]

            rows_sib = cursor.execute(sql_sib, params_sib).fetchall()
            if not rows_sib:
                continue

            C_sib = random.choice(rows_sib)
            sib_id = C_sib["id"]
            sib_narrs = json.loads(C_sib["narrations_json"])

            # 4.2) Costruisci query sibling usando narrations di questo evento DB
            Q_fill = {
                "video_uid": video_uid,  # stessa clip e stesso video dell'anchor
                "matched_template": (tpl_i if not geom_only and tpl_i is not None else "GEOM_ONLY"),
                "narrations": sib_narrs,
                "query_start_sec": float(qs_abs),
                "query_end_sec": float(qe_abs),
                "clip_start_sec": float(clip_cs),
                "clip_end_sec": float(clip_ce),
                "bucket": bucket_i,
            }
            query_piazzate.append(Q_fill)
            clip_occupied.append((qs_abs, qe_abs))

            candidati_usati.add(sib_id)

            fill_key = bucket_i if geom_only else (tpl_i, bucket_i)
            if fill_key in G_plan_remaining and G_plan_remaining[fill_key] > 0:
                G_plan_remaining[fill_key] -= 1
                if G_plan_remaining[fill_key] == 0:
                    del G_plan_remaining[fill_key]
            placed_by_cell[fill_key] += 1
            placed_by_bucket[bucket_i] += 1
            total_targets_remaining -= 1
            pbar.update(1)

        # 5) chiudi il clip e marca tutti i candidates usati (anchor + siblings)
        final_clip_obj = {
            "video_uid": video_uid,
            "clip_start_sec": clip_cs,
            "clip_end_sec": clip_ce,
            "narrations": [],
            "queries": query_piazzate,
        }
        clips_multi.append(final_clip_obj)
        mark_candidates_used(db_conn, candidati_usati)
        clip_attempts = 0

    pbar.close()
    if total_targets_remaining > 0:
        print(f"[WARN] Allocazione terminata, ma {total_targets_remaining} target non piazzati.")
        if clip_attempts >= max_clip_creation_attempts:
            print(f"[WARN] ...Interrotta per troppi fallimenti consecutivi nel creare clip (max: {max_clip_creation_attempts}).")

    return clips_multi, placed_by_cell, placed_by_bucket

# --- Esecuzione ---
clips_multi, placed_by_cell, placed_by_bucket = allocate_holistic_trq_anchor_fill(
    db_conn, G_plan, TRQ, video_max_ts, GEOM_ONLY_MODE
)

db_conn.close()
print("Connessione DB chiusa.")

n_clips  = len(clips_multi)
n_queries = sum(len(c["queries"]) for c in clips_multi)
print(f"\n[RISULTATO] clip={n_clips} | queries={n_queries}")

# ---------- FASE 5/5: Salvataggio e Report ----------
print("\n[FASE 5/5] Salvataggio e generazione report...")

print("Scrivo i contenitori...")
meta = {
    "final_target": int(FINAL_TARGET),
    "safety": float(SAFETY),
    "accept_prior": [float(x) for x in ACC_PRIOR],
    "duration_buckets": [float(x) for x in DURATION_BUCKETS],
    "min_query_len_s": float(MIN_QUERY_LEN_S),
    "allocation_mode": "Holistic-Clen-Centric-DB-TRQAnchorFill",
    "geom_only_mode": GEOM_ONLY_MODE,
    "holistic_max_q_per_clip": HOLISTIC_MAX_Q_PER_CLIP,
    "holistic_clen_tolerance_s": CLEN_TOLERANCE_S,
    "holistic_alignment_tolerance_s": ALIGNMENT_TOLERANCE_S,
    "sampler": "trq_anchor_trq_fill",
    "template_ruling" : "semantic_tagging" if not GEOM_ONLY_MODE else "None",
    "joint_target_cells": int(len(T_joint)),
    "joint_target_sum": int(sum(T_joint.values())),
    "planned_cells": int(len(G_plan)),
    "planned_sum": int(sum(G_plan.values())),
}

out_multi = {"meta": meta, "clips": clips_multi}
os.makedirs(os.path.dirname(OUT_MULTI_PATH), exist_ok=True)
with open(OUT_MULTI_PATH, "w", encoding="utf-8") as f:
    json.dump(out_multi, f, ensure_ascii=False, indent=2)
print(f"[OK] Scritto MULTI: {OUT_MULTI_PATH} | clip={n_clips} | queries={n_queries}")

# --- FLAT + RACCOLTA METRICHE ---
flat = []
gen_durations = []
gen_templates = []
gen_tb_pairs = []

for c in clips_multi:
    all_clip_narrs = []
    for q in c["queries"]:
        all_clip_narrs.extend(q.get("narrations", []))
    
    seen_narr_ids = set()
    unique_narrs = []
    for n in all_clip_narrs:
        nid = n.get("narration_uid", n.get("narration_text", ""))
        if nid not in seen_narr_ids:
            unique_narrs.append(n)
            seen_narr_ids.add(nid)
    try:
        unique_narrs.sort(key=lambda n: float(n.get("timestamp_sec", 0.0)))
    except (ValueError, TypeError):
        pass
    ctx = " ".join(n.get("narration_text", "") for n in unique_narrs)

    for q in c["queries"]:
        d = float(q["query_end_sec"]) - float(q["query_start_sec"])
        gen_durations.append(d)
        tpl = q.get("matched_template")
        gen_templates.append(tpl)
        bucket = q.get("bucket", bucket_of(d))
        gen_tb_pairs.append((tpl, bucket))
        flat.append({
            "video_uid": c["video_uid"],
            "clip_start_sec": float(c["clip_start_sec"]),
            "clip_end_sec":   float(c["clip_end_sec"]),
            "query_start_sec": float(q["query_start_sec"]),
            "query_end_sec":   float(q["query_end_sec"]),
            "matched_template": tpl,
            "narrations": q.get("narrations", []),
            "context_text": ctx
        })

with open(OUT_FLAT_PATH, "w", encoding="utf-8") as f:
    json.dump(flat, f, ensure_ascii=False, indent=2)
print(f"[OK] Scritto FLAT : {OUT_FLAT_PATH}  | rows={len(flat)}")

# --- REPORT (Plot) ---
def ecdf(x):
    x = np.sort(np.asarray(x, dtype=float))
    y = np.arange(1, len(x)+1) / max(1, len(x))
    return x, y

# 1) ECDF durate
train_durations = np.asarray(original_durations, dtype=float)
gen_durations = np.asarray(gen_durations, dtype=float)
if len(gen_durations) > 0:
    plt.figure(figsize=(6,4))
    xt, yt = ecdf(train_durations); xg, yg = ecdf(gen_durations)
    plt.plot(xt, yt, label="train", linewidth=2)
    plt.plot(xg, yg, label="generated", linewidth=2)
    plt.xlabel("Durata [s]"); plt.ylabel("ECDF")
    plt.title("ECDF durate: train vs generated")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(CATALOG_PATH, "clip/ecdf_duration.png"))
    plt.close()
else:
    print("[REPORT] Saltato ECDF durate (0 query generate).")

# 1bis) HISTOGRAM durate
BIN_W = 0.5
edges = np.arange(0, 60 + BIN_W, BIN_W)
plt.figure(figsize=(8,4))
plt.hist(train_durations[(train_durations>0)&(train_durations<60)],
         bins=edges, density=True, alpha=0.45, label="train", histtype="stepfilled")
if len(gen_durations) > 0:
    plt.hist(gen_durations[(gen_durations>0)&(gen_durations<60)],
             bins=edges, density=True, alpha=0.55, label="generated", histtype="stepfilled")
plt.xlabel("Query duration (s)"); plt.ylabel("Density")
plt.title("Duration distribution: train vs generated")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout(); plt.savefig(os.path.join(CATALOG_PATH, "clip/duration_distribution.png"))
plt.close()

# 2) Distribuzione per bucket
def bucket_hist(durs):
    counts = defaultdict(int)
    for d in durs:
        if d>0: counts[bucket_of(float(d))] += 1
    order = [(DURATION_BUCKETS[i], DURATION_BUCKETS[i+1]) for i in range(len(DURATION_BUCKETS)-1)]
    total = sum(counts.values()) or 1
    xs = [f"[{a},{b})" for (a,b) in order]
    ys = [counts.get((a,b),0)/total for (a,b) in order]
    return xs, ys

xs, yt = bucket_hist(train_durations)
_, yg = bucket_hist(gen_durations)
plt.figure(figsize=(8,4))
w = 0.4; idx = np.arange(len(xs))
plt.bar(idx - w/2, yt, width=w, label="train")
plt.bar(idx + w/2, yg, width=w, label="generated")
plt.xticks(idx, xs, rotation=30); plt.ylabel("Frazione")
plt.title("Distribuzione durate per bucket")
plt.legend(); plt.grid(True, axis='y', alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(CATALOG_PATH, "clip/bucket_distribution.png"))
plt.close()

# 3) Template distribution
if not GEOM_ONLY_MODE:
    train_tpl_series = pd.Series(template_distribution).sort_values(ascending=False)
    gen_tpl_series = pd.Series(gen_templates).value_counts(normalize=True)
    all_tpls = list(set(train_tpl_series.index).union(set(gen_tpl_series.index)))
    train_tpl_aligned = train_tpl_series.reindex(all_tpls).fillna(0.0)
    gen_tpl_aligned = gen_tpl_series.reindex(all_tpls).fillna(0.0)
    topN = 20
    top_idx = train_tpl_aligned.sort_values(ascending=False).index[:topN]
    plt.figure(figsize=(10,5))
    w = 0.4; idx = np.arange(len(top_idx))
    plt.bar(idx - w/2, train_tpl_aligned[top_idx].values, width=w, label="train")
    plt.bar(idx + w/2, gen_tpl_aligned[top_idx].values, width=w, label="generated")
    plt.xticks(idx, [str(t)[:24] + ("…" if len(str(t))>24 else "") for t in top_idx], rotation=45, ha='right')
    plt.ylabel("Frazione"); plt.title("Template distribution (top 20)")
    plt.legend(); plt.grid(True, axis='y', alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(CATALOG_PATH, "clip/template_distribution.png"))
    plt.close()
else:
    print("[REPORT] Saltato plot Template (modalità --geom_only).")

# 4) Matching T×B
if not GEOM_ONLY_MODE:
    tj_df = pd.Series(build_joint_budget(train_json, FINAL_TARGET, DURATION_BUCKETS, geom_only=False)).rename("target")
    tj_df = tj_df[tj_df>0]; tj_df = tj_df / max(1, tj_df.sum())
    gen_pairs = pd.Series(gen_tb_pairs).value_counts(); gen_pairs = gen_pairs / max(1, gen_pairs.sum())
    all_keys = set(tj_df.index).union(set(gen_pairs.index))
    t_vals, g_vals, labels = [], [], []
    for k in all_keys:
        t_vals.append(float(tj_df.get(k, 0.0)))
        g_vals.append(float(gen_pairs.get(k, 0.0)))
        tpl, b = k; labels.append(f"{str(tpl)[:14]}|[{b[0]},{b[1]})")
    order_idx = np.argsort(-np.array(t_vals)); topM = 25; sel = order_idx[:topM]
    plt.figure(figsize=(12,6))
    w = 0.4; idx = np.arange(len(sel))
    plt.bar(idx - w/2, np.array(t_vals)[sel], width=w, label="target(train proxy)")
    plt.bar(idx + w/2, np.array(g_vals)[sel], width=w, label="generated")
    plt.xticks(idx, [labels[i] for i in sel], rotation=60, ha='right')
    plt.ylabel("Frazione"); plt.title("Template×Bucket matching (top 25)")
    plt.legend(); plt.grid(True, axis='y', alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(CATALOG_PATH, "clip/matching_template.png"))
    plt.close()
else:
    print("[REPORT] Saltato plot Matching T×B (modalità --geom_only).")


# KS relpos per bucket

print("\n[REPORT] Calcolo KS relpos per bucket (multi-query) ...")
gen_rows = []
for c in clips_multi:
    cs, ce = float(c["clip_start_sec"]), float(c["clip_end_sec"])
    clen = max(1e-9, ce - cs)
    for q in c["queries"]:
        s = float(q["query_start_sec"]); e = float(q["query_end_sec"])
        d = e - s
        if clen <= 0: continue
        qs_rel = (s - cs)/clen
        qe_rel = (e - cs)/clen
        if not (0.0 <= qs_rel <= 1.0 and 0.0 <= qe_rel <= 1.0 and qs_rel < qe_rel):
            continue 
        gen_rows.append({
            "video_uid": c["video_uid"],
            "clip_cs": cs, "clip_ce": ce, "clen": clen,
            "qs": s, "qe": e, "d": d, "qs_rel": qs_rel, "qe_rel": qe_rel,
            "bucket": q.get("bucket", bucket_of(float(d)))
        })
GEQ = pd.DataFrame(gen_rows)
TRQ_1 = TRQ[TRQ["d"].notna()].copy()
GEQ_1 = GEQ.copy() 

BEDS = DURATION_BUCKETS
ks_rows = []
for lo,hi in zip(BEDS[:-1], BEDS[1:]):
    b = (float(lo), float(hi))
    a = TRQ_1[TRQ_1["bucket"]==b]["qs_rel"].to_numpy() if "bucket" in TRQ_1.columns else np.array([])
    c = GEQ_1[GEQ_1["bucket"]==b]["qs_rel"].to_numpy() if "bucket" in GEQ_1.columns else np.array([])
    ks_qs = ks_2samp(a, c).statistic if (len(a)>1 and len(c)>1) else np.nan
    a = TRQ_1[TRQ_1["bucket"]==b]["qe_rel"].to_numpy() if "bucket" in TRQ_1.columns else np.array([])
    c = GEQ_1[GEQ_1["bucket"]==b]["qe_rel"].to_numpy() if "bucket" in GEQ_1.columns else np.array([])
    ks_qe = ks_2samp(a, c).statistic if (len(a)>1 and len(c)>1) else np.nan
    ks_rows.append({"bucket": b, "KS(qs_rel)": ks_qs, "KS(qe_rel)": ks_qe})

df_ks = pd.DataFrame(ks_rows)
print("\n[REPORT] KS per bucket (multi-query) — più basso è meglio:")
print(df_ks)

# --- Plot KS (qs_rel) ---
buckets_list = [(DURATION_BUCKETS[i], DURATION_BUCKETS[i+1]) for i in range(len(DURATION_BUCKETS)-1)]

fig, axs = plt.subplots(2, 3, figsize=(12,6), sharey=True)
axs = axs.ravel()
for i, b in enumerate(buckets_list):
    ax = axs[i]
    a = TRQ_1[TRQ_1["bucket"]==b]["qs_rel"].to_numpy() if "bucket" in TRQ_1.columns else np.array([])
    c = GEQ_1[GEQ_1["bucket"]==b]["qs_rel"].to_numpy() if "bucket" in GEQ_1.columns else np.array([])
    if len(a)+len(c)==0:
        ax.set_title(f"[{b[0]}, {b[1]}) — (vuoto)"); ax.axis('off'); continue
    if len(a)>0: ax.hist(a, bins=np.linspace(0, 1, 21), density=True, alpha=0.45, label="train", histtype="stepfilled")
    if len(c)>0: ax.hist(c, bins=np.linspace(0, 1, 21), density=True, alpha=0.55, label="generated", histtype="stepfilled")
    ax.set_title(f"qs_rel [{b[0]}, {b[1]})")
    ax.grid(True, linestyle="--", alpha=0.3)
    if i%3==0: ax.set_ylabel("Density")
    ax.set_xlabel("qs_rel")
if len(gen_durations) > 0:
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
fig.suptitle("Posizioni relative — qs_rel per bucket (train vs generated)")
plt.tight_layout(rect=[0,0,0.96,0.95]); plt.savefig(os.path.join(CATALOG_PATH, "clip/qs_rel.png"))
plt.close(fig)

# --- Plot KS (qe_rel) ---
fig, axs = plt.subplots(2, 3, figsize=(12,6), sharey=True)
axs = axs.ravel()
for i, b in enumerate(buckets_list):
    ax = axs[i]
    a = TRQ_1[TRQ_1["bucket"]==b]["qe_rel"].to_numpy() if "bucket" in TRQ_1.columns else np.array([])
    c = GEQ_1[GEQ_1["bucket"]==b]["qe_rel"].to_numpy() if "bucket" in GEQ_1.columns else np.array([])
    if len(a)+len(c)==0:
        ax.set_title(f"[{b[0]}, {b[1]}) — (vuoto)"); ax.axis('off'); continue
    if len(a)>0: ax.hist(a, bins=np.linspace(0, 1, 21), density=True, alpha=0.45, label="train", histtype="stepfilled")
    if len(c)>0: ax.hist(c, bins=np.linspace(0, 1, 21), density=True, alpha=0.55, label="generated", histtype="stepfilled")
    ax.set_title(f"qe_rel [{b[0]}, {b[1]})")
    ax.grid(True, linestyle="--", alpha=0.3)
    if i%3==0: ax.set_ylabel("Density")
    ax.set_xlabel("qe_rel")
if len(gen_durations) > 0:
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
fig.suptitle("Posizioni relative — qe_rel per bucket (train vs generated)")
plt.tight_layout(rect=[0,0,0.96,0.95]); plt.savefig(os.path.join(CATALOG_PATH, "clip/qe_rel.png"))
plt.close(fig)

# --- Plot KS (u_from_df) ---
def u_from_df(df):
    w = (df["qe_rel"] - df["qs_rel"]).to_numpy()
    denom = np.clip(1.0 - w, 1e-6, 1.0)
    u = (df["qs_rel"] / denom).astype(float)
    return u[(u >= 0.0) & (u <= 1.0)]

def ecdf_vals(x):
    x = np.sort(np.asarray(x, float))
    y = np.arange(1, len(x)+1) / max(1, len(x))
    return x, y

ks_u_rows = []
fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharey=True, sharex=True)
axs = axs.ravel()
for i, b in enumerate(buckets_list):
    ax = axs[i]
    u_train = u_from_df(TRQ_1[TRQ_1["bucket"] == b]) if "bucket" in TRQ_1.columns else np.array([])
    u_gen   = u_from_df(GEQ_1[GEQ_1["bucket"] == b]) if "bucket" in GEQ_1.columns else np.array([])
    if len(u_train) + len(u_gen) == 0:
        ax.set_title(f"u ECDF [{b[0]}, {b[1]}) — vuoto"); ax.axis('off')
        ks_u_rows.append({"bucket": b, "KS(u_train|u_gen)": np.nan, "n_train": 0, "n_gen": 0})
        continue
    xt, yt = ecdf_vals(u_train); xg, yg = ecdf_vals(u_gen)
    if len(xt) > 0: ax.plot(xt, yt, label="train", linewidth=2)
    if len(xg) > 0: ax.plot(xg, yg, label="generated", linewidth=2, linestyle='--')
    ax.set_title(f"u ECDF [{b[0]}, {b[1]})")
    ax.grid(True, alpha=0.3)
    if i % 3 == 0: ax.set_ylabel("ECDF")
    if i >= 3: ax.set_xlabel("u = qs_rel / (1 - w)")
    ks_stat = ks_2samp(u_train, u_gen).statistic if (len(u_train)>1 and len(u_gen)>1) else np.nan
    ks_u_rows.append({"bucket": b, "KS(u_train|u_gen)": float(ks_stat), "n_train": len(u_train), "n_gen": len(u_gen)})

if len(gen_durations) > 0:
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
fig.suptitle("Confronto ECDF di 'u': Train vs Generated per Bucket")
plt.tight_layout(rect=[0, 0, 0.96, 0.95]); plt.savefig(os.path.join(CATALOG_PATH, "clip/report_ecdf_u.png"))
plt.close(fig)

df_ks_u = pd.DataFrame(ks_u_rows)
print("\n[VERIFICA] KS(u_train|u_gen) — Confronto diretto non parametrico. Più basso è meglio:")
print(df_ks_u.to_string(index=False))
print("\n[REPORT] Completato.")
