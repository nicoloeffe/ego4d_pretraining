# ----------------------------------------------------------------------
# SCRIPT 1: create_catalog_db.py
#
# Scopo: Generare un db contenente "scene" di interesse .
#        Estrae tutti i candidati-evento e li salva in un database
#        SQLite, escludendo finestre che contengono "#unsure".
#
# Uso (Debug Veloce):
# > python create_catalog_db.py --geom_only
#
# Uso (Produzione):
# > python create_catalog_db.py
# ----------------------------------------------------------------------

import os, json, math, random, bisect, argparse, sqlite3, gc
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm
import torch 
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
BASE_PATH = ""
NARR_JSON_PATH  = os.path.join(BASE_PATH, "narration.json")
CATALOG_PATH    = "pretrain_catalog/"
CATALOG_DB_PATH = os.path.join(CATALOG_PATH, "catalog.db")

# File JSON (servono solo per i nomi dei template e gli split)
TRAIN_JSON_PATH = os.path.join(BASE_PATH, "nlq_train.json")
VAL_JSON_PATH = "ego4d_data/v1/annotations/nlq_val.json"
TEST_JSON_PATH = "ego4d_data/v1/annotations/nlq_test_unannotated.json"
PROTOTYPES_PATH = "template_prototypes.npy" 
TEMPLATE_NAMES_PATH = "template_names.json" 

# Config Generazione 
SHORT_THR       = 4.0
EVENT_GAP_S     = 8.0
DURATION_BUCKETS = [0,2,4,8,16,32,60]
TARGET_VIDEO_COUNT = 1000

ROLL_LONG_ENABLE            = True
ROLL_LONG_LENGTHS_S         = [36,44,52,60,68]
ROLL_LONG_BASE_STEP_S       = 8.0
ROLL_LONG_ADAPTIVE_FRAC     = 0.08
ROLL_LONG_MAX_WINDOWS_PER_V = 4000
ROLL_LONG_MIN_NARR          = 4
ROLL_LONG_MAX_VIDEO_LEN_S   = 6*3600
NARR_EPS = 1e-3

# Config Semantica 
MODEL_NAME_SEMANTIC = "sentence-transformers/all-mpnet-base-v2"
CATALOG_ASSIGN_TOP_K = 5
SEMANTIC_ENCODE_BATCH_SIZE = 256

DB_WRITE_BATCH_SIZE = 1000 

print(f"--- Inizializzazione Creazione Catalogo DB ---")
print(f"Percorso DB: {CATALOG_DB_PATH}")


parser = argparse.ArgumentParser(description="Script per creare il database del catalogo (catalog.db).")
parser.add_argument(
    "--geom_only",
    action="store_true",
    help="Modalità Debug Veloce: Salta l'encoding semantico e salva solo i dati geometrici."
)
args = parser.parse_args()
GEOM_ONLY_MODE = args.geom_only

if GEOM_ONLY_MODE:
    print("[ATTENZIONE] Esecuzione in modalità --geom_only (Debug Veloce).")
    print("L'analisi semantica (template) verrà saltata.")
else:
    print("[INFO] Esecuzione in modalità Produzione (Completa).")
    print("Verrà eseguita l'analisi semantica (lento).")

rng = np.random.default_rng(1234)
random.seed(1234)

# ---------- UTILS ----------
def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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

def segment_events(narr_list, gap_s=EVENT_GAP_S):
    evts = []
    if not narr_list: return evts
    narr_list = sorted(narr_list, key=lambda n: float(n['timestamp_sec']))
    cur = [narr_list[0]]
    for n in narr_list[1:]:
        if (float(n['timestamp_sec']) - float(cur[-1]['timestamp_sec'])) > gap_s:
            if len(cur) > 1: evts.append(cur)
            cur = []
        cur.append(n)
    if len(cur) > 1: evts.append(cur)
    return evts

def choose_source_videos(pretrain_narrations, target_video_count):
    stats = []
    for vid, data in pretrain_narrations.items():
        narr = data.get("narration_pass_1", {}).get("narrations", [])
        stats.append((vid, len(narr)))
    stats.sort(key=lambda x: x[1], reverse=True)
    chosen = [vid for vid, cnt in stats if cnt >= 2][:target_video_count]
    return set(chosen)

def get_split_video_uids(json_path):
    if not os.path.exists(json_path):
        print(f"[WARN] File split non trovato: {json_path}. Verrà ignorato.")
        return set()
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'videos' in data:
             return {v['video_uid'] for v in data['videos']}
        else:
             print(f"[WARN] Formato inatteso: {json_path}.")
             return set()
    except Exception as e:
        print(f"[ERROR] Errore lettura {json_path}: {e}")
        return set()

def get_all_narrations_by_video(narr_json_path):
    if not os.path.exists(narr_json_path):
        raise FileNotFoundError(narr_json_path)
    with open(narr_json_path, 'r') as f:
        return json.load(f)

# ---------- Gestione DB ----------
def init_db(db_path):
    """Crea (o resetta) il file DB e le tabelle."""
    print(f"Inizializzazione database a: {db_path}")
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # --- Tabella 1: Geometria ---
    cursor.execute("""
    CREATE TABLE candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_uid TEXT NOT NULL,
        cs REAL NOT NULL,
        ce REAL NOT NULL,
        _len REAL NOT NULL,
        bucket_lo REAL NOT NULL,
        bucket_hi REAL NOT NULL,
        narrations_json TEXT,
        usato INTEGER DEFAULT 0 
    )
    """)
    
    # --- Tabella 2: Template  ---
    if not GEOM_ONLY_MODE:
        cursor.execute("""
        CREATE TABLE templates (
            candidate_id INTEGER NOT NULL,
            template_name TEXT NOT NULL,
            FOREIGN KEY (candidate_id) REFERENCES candidates (id)
        )
        """)
    
    print("Schema DB creato.")
    conn.commit()
    return conn

def create_db_indexes(conn):
    """Crea gli indici alla FINE, è molto più veloce."""
    print("Creazione indici DB (richiede tempo)...")
    
    # Indice per cercare (TPL, Len) - per Q1
    if not GEOM_ONLY_MODE:
        print("... Indice: (template_name, _len)")
        conn.execute("CREATE INDEX idx_tpl_len ON templates (template_name, candidate_id)")
        conn.execute("CREATE INDEX idx_cand_len ON candidates (_len, id)") # Aiuta il join
    
    # Indice per cercare (TPL, VID, Len) - per Qk
    if not GEOM_ONLY_MODE:
        print("... Indice: (template_name, video_uid, _len)")
        conn.execute("CREATE INDEX idx_cand_vid_len ON candidates (video_uid, _len, id)")
        
    # Indice per il debug --geom_only
    if GEOM_ONLY_MODE:
        print("... Indice --geom_only: (bucket_lo, bucket_hi, _len)")
        conn.execute("CREATE INDEX idx_geom_bucket_len ON candidates (bucket_lo, bucket_hi, _len, id)")
        print("... Indice --geom_only: (bucket_lo, bucket_hi, video_uid, _len)")
        conn.execute("CREATE INDEX idx_geom_bucket_vid_len ON candidates (bucket_lo, bucket_hi, video_uid, _len, id)")

    conn.commit()
    print("Indici DB creati.")

def write_batch_to_db(
    db_conn, 
    temp_data_list, 
    texts_to_encode, 
    geom_only,
    model, 
    prototypes, 
    template_names
    ):
    """
    Funzione helper che prende un batch, esegue l'encoding (se non geom_only)
    e scrive i risultati nel database.
    """
    if not temp_data_list:
        return 0

    try:
        candidate_rows = []
        
        
        if geom_only:
            for temp_data in temp_data_list:
                bucket = temp_data["bucket"]
                candidate_rows.append((
                    temp_data["video_uid"], temp_data["base_start_sec"], temp_data["base_end_sec"],
                    temp_data["_len"], bucket[0], bucket[1],
                    json.dumps(temp_data["narrations"])
                ))
        
        else:
            if not texts_to_encode or model is None or prototypes is None:
                 print("[WARN] Dati di encoding mancanti in modalità Produzione. Batch saltato.")
                 return 0

            embeddings = model.encode(
                texts_to_encode, batch_size=SEMANTIC_ENCODE_BATCH_SIZE,
                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            
            all_scores = embeddings @ prototypes.T
            k_check = min(CATALOG_ASSIGN_TOP_K, len(template_names))
            top_k_indices_batch = np.argsort(all_scores, axis=1)[:, -k_check:][:, ::-1]

            for i, temp_data in enumerate(temp_data_list):
                bucket = temp_data["bucket"]
                candidate_rows.append((
                    temp_data["video_uid"], temp_data["base_start_sec"], temp_data["base_end_sec"],
                    temp_data["_len"], bucket[0], bucket[1],
                    json.dumps(temp_data["narrations"])
                ))
                
                valid_indices = [idx for idx in top_k_indices_batch[i] if 0 <= idx < len(template_names)]
                top_k_templates = [template_names[j] for j in valid_indices]
                temp_data['potential_templates'] = top_k_templates 

        # --- Scrittura ---
        cursor = db_conn.cursor()
        
        if geom_only:
            cursor.executemany("""
            INSERT INTO candidates (video_uid, cs, ce, _len, bucket_lo, bucket_hi, narrations_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, candidate_rows)
        
        else:
            for i, c_row in enumerate(candidate_rows):
                cursor.execute("""
                INSERT INTO candidates (video_uid, cs, ce, _len, bucket_lo, bucket_hi, narrations_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, c_row)
                
                candidate_id = cursor.lastrowid
                
                tpl_rows_for_this_id = [
                    (candidate_id, tpl_name) 
                    for tpl_name in temp_data_list[i]['potential_templates']
                ]
                if tpl_rows_for_this_id:
                    cursor.executemany("""
                    INSERT INTO templates (candidate_id, template_name)
                    VALUES (?, ?)
                    """, tpl_rows_for_this_id)

        db_conn.commit()
        return len(candidate_rows)

    except torch.cuda.OutOfMemoryError as oom_err:
         print(f"ERROR: CUDA Out Of Memory! {oom_err}")
         raise oom_err
    except Exception as e:
        print(f"ERROR durante scrittura batch DB: {e}")
        db_conn.rollback() 
        return 0

# ---  Funzioni di Catalogo  ---
def make_catalog_short(db_conn, geom_only, pool_videos, all_narrs, model, prototypes, template_names):
    texts_to_encode = []
    temp_data_list = []
    total_added = 0
    
    print("Inizio Catalogo 'Short'...")
    pbar = tqdm(pool_videos, desc="Catalogo short", leave=False)
    for vid in pbar:
        data = all_narrs.get(vid); narr = data.get("narration_pass_1", {}).get("narrations", []) if data else []
        valid_narrs = [n for n in narr if n.get("timestamp_sec") is not None]
        if len(valid_narrs) < 2: continue
        try: valid_narrs.sort(key=lambda n: float(n["timestamp_sec"]))
        except (ValueError, TypeError): continue

        for i in range(len(valid_narrs) - 1):
            for w in (2, 3):
                if i + w > len(valid_narrs): continue
                sub = valid_narrs[i:i + w]
                try: s, e = float(sub[0]['timestamp_sec']), float(sub[-1]['timestamp_sec'])
                except (ValueError, TypeError, KeyError, IndexError): continue
                dur = e - s
                if 1e-6 < dur <= SHORT_THR:
                    
                    # --- Rimuovo gli #unsure ---
                    contains_unsure = False
                    all_texts = []
                    for n in sub:
                        narration_text = n.get('narration_text', '')
                        if "#unsure" in narration_text:
                            contains_unsure = True
                            break
                        all_texts.append(narration_text)
                    
                    if contains_unsure:
                        continue # Salta questa finestra
                    
                    txt = " ".join(all_texts).strip()
                    # --- Fine Modifica ---
                    
                    if txt or geom_only:
                         texts_to_encode.append(txt if txt else "dummy text")
                         temp_data_list.append({
                             "video_uid": vid, "narrations": sub,
                             "base_start_sec": s, "base_end_sec": e,
                             "bucket": bucket_of(dur),
                             "_len": dur
                         })
        
        # --- Scrittura Batch ---
        if len(texts_to_encode) >= DB_WRITE_BATCH_SIZE:
            added = write_batch_to_db(db_conn, temp_data_list, texts_to_encode, geom_only,
                                      model, prototypes, template_names)
            total_added += added
            pbar.set_postfix(total=total_added)
            texts_to_encode, temp_data_list = [], []

    # --- Scrittura reminder ---
    if texts_to_encode:
        added = write_batch_to_db(db_conn, temp_data_list, texts_to_encode, geom_only,
                                  model, prototypes, template_names)
        total_added += added
    
    print(f"Catalogo 'Short' completato: {total_added} candidati scritti su DB.")
    return total_added

def make_catalog_long(db_conn, geom_only, pool_videos, all_narrs, model, prototypes, template_names):
    texts_to_encode = []
    temp_data_list = []
    total_added = 0
    
    print("Inizio Catalogo 'Long' (Eventi)...")
    pbar = tqdm(pool_videos, desc="Catalogo long (Eventi)", leave=False)
    for vid in pbar:
        data = all_narrs.get(vid); narr = data.get("narration_pass_1", {}).get("narrations", []) if data else []
        valid_narrs = [n for n in narr if n.get("timestamp_sec") is not None]
        if len(valid_narrs) < 2: continue
        try:
             valid_narrs.sort(key=lambda n: float(n["timestamp_sec"]))
             evts = segment_events(valid_narrs, gap_s=EVENT_GAP_S)
        except (ValueError, TypeError): continue

        for ev in evts:
            if len(ev) < 2: continue
            for w in (3, 4, 5, 6, 7, 8):
                 for i in range(len(ev) - w + 1):
                    sub = ev[i : i + w]
                    if len(sub) != w: continue 
                    try: s_narr, e_narr = float(sub[0]['timestamp_sec']), float(sub[-1]['timestamp_sec'])
                    except (ValueError, TypeError, KeyError, IndexError): continue
                    dur = e_narr - s_narr
                    if dur < SHORT_THR or dur >= DURATION_BUCKETS[-1]: continue
                    
                    # --- Rimuovo gli #unsure ---
                    contains_unsure = False
                    all_texts = []
                    for n in sub:
                        narration_text = n.get('narration_text', '')
                        if "#unsure" in narration_text:
                            contains_unsure = True
                            break
                        all_texts.append(narration_text)
                    
                    if contains_unsure:
                        continue # Salta questa finestra
                    
                    txt = " ".join(all_texts).strip()

                    if txt or geom_only:
                         texts_to_encode.append(txt if txt else "dummy text")
                         temp_data_list.append({
                             "video_uid": vid, "narrations": sub,
                             "base_start_sec": s_narr, "base_end_sec": e_narr,
                             "bucket": bucket_of(dur),
                             "_len": dur
                         })

        # --- Scrittura Batch ---
        if len(texts_to_encode) >= DB_WRITE_BATCH_SIZE:
            added = write_batch_to_db(db_conn, temp_data_list, texts_to_encode, geom_only,
                                      model, prototypes, template_names)
            total_added += added
            pbar.set_postfix(total=total_added)
            texts_to_encode, temp_data_list = [], []

    # --- Scrittura reminder ---
    if texts_to_encode:
        added = write_batch_to_db(db_conn, temp_data_list, texts_to_encode, geom_only,
                                  model, prototypes, template_names)
        total_added += added

    print(f"Catalogo 'Long' completato: {total_added} candidati scritti su DB.")
    return total_added

def make_catalog_rolling_long(db_conn, geom_only, pool_videos, all_narrs, model, prototypes, template_names):
    if not ROLL_LONG_ENABLE: return 0
    texts_to_encode = []
    temp_data_list = []
    total_added = 0
    
    print("Inizio Catalogo 'Rolling Long'...")
    pbar = tqdm(pool_videos, desc="Rolling long windows", leave=False)
    for vid in pbar:
        data = all_narrs.get(vid); narr = data.get("narration_pass_1", {}).get("narrations", []) if data else []
        valid_narrs = [n for n in narr if n.get("timestamp_sec") is not None]
        if len(valid_narrs) < ROLL_LONG_MIN_NARR: continue
        try:
            valid_narrs.sort(key=lambda n: float(n["timestamp_sec"]))
            times = np.array([float(n["timestamp_sec"]) for n in valid_narrs], dtype=float)
            if times.size == 0: continue
            t0, t1 = float(times[0]), float(times[-1]); Vlen = max(0.0, t1 - t0)
        except (ValueError, TypeError, IndexError): continue
        if Vlen <= 1e-6 or Vlen > ROLL_LONG_MAX_VIDEO_LEN_S: continue

        base_step = ROLL_LONG_BASE_STEP_S
        adapt_step = max(base_step, Vlen * ROLL_LONG_ADAPTIVE_FRAC) if ROLL_LONG_LENGTHS_S else base_step
        adapt_step = float(min(adapt_step, 2.0 * max(ROLL_LONG_LENGTHS_S))) if ROLL_LONG_LENGTHS_S and ROLL_LONG_LENGTHS_S else base_step
        windows_added_here = 0; max_win_here = ROLL_LONG_MAX_WINDOWS_PER_V

        for L in ROLL_LONG_LENGTHS_S:
            if windows_added_here >= max_win_here or L <= 0: continue
            step = max(base_step, min(adapt_step, 0.5 * L)); step = max(1e-6, step)
            start = t0
            try: num_steps_approx = math.ceil(Vlen / step) if step > 0 else 1; hard_cap_calc = min(max_win_here - windows_added_here, num_steps_approx + 2); hard_cap = max(1, int(hard_cap_calc))
            except OverflowError: hard_cap = max_win_here
            it = 0; current_start = start
            while current_start + L <= t1 + NARR_EPS and windows_added_here < max_win_here and it < hard_cap:
                end = current_start + L
                i0 = bisect.bisect_left(times, current_start - NARR_EPS); i1 = bisect.bisect_right(times, end + NARR_EPS)
                if (i1 - i0) >= ROLL_LONG_MIN_NARR:
                    sub = valid_narrs[i0:i1]
                    if sub:
                        
                        
                        contains_unsure = False
                        all_texts = []
                        for n in sub:
                            narration_text = n.get('narration_text', '')
                            if "#unsure" in narration_text:
                                contains_unsure = True
                                break
                            all_texts.append(narration_text)
                        
                        if contains_unsure:
                            current_start += step; it += 1 
                            continue # Salta questa finestra
                        
                        txt = " ".join(all_texts).strip()
                        
                        
                        if txt or geom_only:
                             texts_to_encode.append(txt if txt else "dummy text")
                             temp_data_list.append({
                                 "video_uid": vid, "narrations": sub,
                                 "base_start_sec": float(current_start), "base_end_sec": float(end),
                                 "bucket": bucket_of(L),
                                 "_len": L
                             })
                             windows_added_here += 1
                current_start += step; it += 1
        
        # --- Scrittura Batch (per video, per efficienza) ---
        if len(texts_to_encode) >= DB_WRITE_BATCH_SIZE:
            added = write_batch_to_db(db_conn, temp_data_list, texts_to_encode, geom_only,
                                      model, prototypes, template_names)
            total_added += added
            pbar.set_postfix(total=total_added)
            texts_to_encode, temp_data_list = [], []

    # --- Scrittura reminder ---
    if texts_to_encode:
        added = write_batch_to_db(db_conn, temp_data_list, texts_to_encode, geom_only,
                                  model, prototypes, template_names)
        total_added += added

    print(f"Catalogo 'Rolling Long' completato: {total_added} candidati scritti su DB.")
    return total_added

def main():
    

    print("[FASE 1/3] Caricamento risorse (Narrazioni, Split Video)...")
    
    try:
        all_narrations = get_all_narrations_by_video(NARR_JSON_PATH)
    except FileNotFoundError:
        print(f"ERRORE: File narrazioni non trovato: {NARR_JSON_PATH}")
        return

    try:
        train_json_file = open(TRAIN_JSON_PATH, 'r')
        train_data = json.load(train_json_file)
        train_video_uids = {v['video_uid'] for v in train_data['videos']}
        train_json_file.close()
        del train_data 
    except FileNotFoundError:
        print(f"ERRORE: File train JSON non trovato: {TRAIN_JSON_PATH}")
        return
        
    val_video_uids = get_split_video_uids(VAL_JSON_PATH)
    test_video_uids = get_split_video_uids(TEST_JSON_PATH)
    exclude_video_uids = train_video_uids.union(val_video_uids).union(test_video_uids)
    
    print(f"Narrazioni caricate: {len(all_narrations)} video")
    print(f"UID da escludere: {len(exclude_video_uids)} (train={len(train_video_uids)}, val={len(val_video_uids)}, test={len(test_video_uids)})")

    pool = {uid: data for uid, data in all_narrations.items() if uid not in exclude_video_uids}
    chosen = choose_source_videos(pool, TARGET_VIDEO_COUNT)
    print(f"Pool disgiunto={len(pool)} | Useremo top-K video={len(chosen)}")

    # --- 2. Caricamento Risorse Semantiche  ---
    semantic_model = None
    template_prototypes = None
    template_names_list = []

    if not GEOM_ONLY_MODE:
        print("[FASE 2/3] Caricamento Risorse Semantiche (Modello, Prototipi)...")
        try:
            print("-> Caricamento modello Sentence Transformer...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            semantic_model = SentenceTransformer(MODEL_NAME_SEMANTIC, device=device) 
            print(f"-> Modello '{MODEL_NAME_SEMANTIC}' caricato su '{device}'.")
            expected_dim = semantic_model.get_sentence_embedding_dimension()

            print(f"-> Caricamento prototipi da: {PROTOTYPES_PATH}")
            template_prototypes = np.load(PROTOTYPES_PATH)
            if template_prototypes.shape[1] != expected_dim:
                 raise ValueError(f"Dimensione embedding prototipi ({template_prototypes.shape[1]}) non corrisponde a quella del modello ({expected_dim})!")
            print(f"-> Prototipi caricati: shape {template_prototypes.shape}")

            print(f"-> Caricamento nomi template da: {TEMPLATE_NAMES_PATH}")
            with open(TEMPLATE_NAMES_PATH, 'r') as f:
                template_names_list = json.load(f)
            if len(template_names_list) != template_prototypes.shape[0]:
                 raise ValueError(f"Numero nomi template ({len(template_names_list)}) non corrisponde al numero di prototipi ({template_prototypes.shape[0]})!")
            print(f"-> Nomi template caricati: {len(template_names_list)} nomi.")
        
        except FileNotFoundError as e:
            print(f"ERRORE: File di risorse semantiche mancante: {e}")
            print("Impossibile continuare in modalità Produzione senza questi file.")
            return
        except Exception as e:
            print(f"ERRORE durante il caricamento delle risorse semantiche: {e}")
            return
    else:
        print("[FASE 2/3] Modalità --geom_only: Caricamento Risorse Semantiche SKIPPATO.")

    # --- 3. Creazione DB e Generazione Catalogo ---
    print(f"[FASE 3/3] Creazione/Reset Database e Popolamento Catalogo...")
    
    db_conn = None
    try:
        db_conn = init_db(CATALOG_DB_PATH)
        
        total_s = make_catalog_short(db_conn, GEOM_ONLY_MODE, chosen, pool, 
                                     semantic_model, template_prototypes, template_names_list)
        force_cleanup()
        
        total_l = make_catalog_long(db_conn, GEOM_ONLY_MODE, chosen, pool, 
                                    semantic_model, template_prototypes, template_names_list)
        force_cleanup()
        total_rl = make_catalog_rolling_long(db_conn, GEOM_ONLY_MODE, chosen, pool, 
                                             semantic_model, template_prototypes, template_names_list)
        force_cleanup()
        
        print("\n--- Generazione Catalogo Completata ---")
        print(f"Candidati 'Short': {total_s}")
        print(f"Candidati 'Long':  {total_l}")
        print(f"Candidati 'Roll':  {total_rl}")
        print(f"TOTALE:            {total_s + total_l + total_rl}")

        # --- 4. Creazione Indici ---
        create_db_indexes(db_conn)

        print(f"\n[SUCCESSO] Database '{CATALOG_DB_PATH}' creato con successo.")
        
    except Exception as e:
        print(f"\n[ERRORE FATALE] Errore durante la creazione del DB: {e}")
    finally:
        if db_conn:
            db_conn.close()
            print("Connessione DB chiusa.")


if __name__ == "__main__":
    main()