import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from collections import defaultdict

BASE_PATH = ""  # Directory base 
TRAIN_JSON_PATH = os.path.join(BASE_PATH, "nlq_train.json")
CATALOG_PATH    = "pretrain_catalog/"
GENERATED_JSON_PATH = os.path.join(CATALOG_PATH, "candidate/pretrain_catalog_v2_multi.json")
OUTPUT_IMG_DIR  = os.path.join(CATALOG_PATH, "plots_ecdf") # Cartella output grafici

DURATION_BUCKETS = [0, 2, 4, 8, 16, 32, 60]


os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)


def canonical_bucket(b):
    """Normalizza la rappresentazione del bucket in una tupla (float, float)."""
    try:
        if isinstance(b, (tuple, list, np.ndarray)):
            return (float(b[0]), float(b[1]))
        return (float(b[0]), float(b[1]))
    except:
        return (0.0, 0.0)

def bucket_of(d, edges=DURATION_BUCKETS):
    """Restituisce il bucket canonico per una data durata d."""
    for i in range(len(edges)-1):
        if edges[i] <= d < edges[i+1]:
            return canonical_bucket((edges[i], edges[i+1]))
    return canonical_bucket((edges[-2], edges[-1]))

def _dedup_segments(pairs, tol=0.05):
    """Rimuove duplicati quasi identici dalle annotazioni di training."""
    if not pairs: return []
    # Ordina e assicura che siano float
    pairs = sorted([(float(s), float(e)) for (s,e) in pairs if s is not None and e is not None and e > s])
    out = []
    for s,e in pairs:
        if not out: 
            out.append([s,e])
            continue
        ps, pe = out[-1]
        # Se c'è overlap quasi totale, fonde (media)
        if abs(s-ps) <= tol and abs(e-pe) <= tol:
            out[-1][0] = 0.5*(ps+s)
            out[-1][1] = 0.5*(pe+e)
        else:
            out.append([s,e])
    return [(s,e) for s,e in out]

def ecdf(x):
    """Calcola la ECDF per il plotting."""
    x = np.sort(np.asarray(x, dtype=float))
    y = np.arange(1, len(x)+1) / max(1, len(x))
    return x, y

def build_TRQ_from_train(train_obj):
    """Costruisce il DataFrame TRQ (Train Relative Queries) dal JSON di training."""
    rows = []
    for v in train_obj["videos"]:
        # vid = v.get("video_uid") # Non strettamente necessario per l'analisi globale
        for c in v.get("clips", []):
            cs = float(c["clip_start_sec"])
            ce = float(c["clip_end_sec"])
            clen = max(1e-9, ce - cs)
            
            all_pairs = []
            # Raccoglie tutte le query nella clip
            for a in c.get("annotations", []):
                for q in a.get("language_queries", []):
                    s = q.get("clip_start_sec")
                    e = q.get("clip_end_sec")
                    # Gestione fallback coordinate video -> clip
                    if s is None or e is None:
                        sv = q.get("video_start_sec")
                        ev = q.get("video_end_sec")
                        cvs = c.get("video_start_sec")
                        if sv is None or ev is None or cvs is None: continue
                        s = (float(sv) - float(cvs)) + cs
                        e = (float(ev) - float(cvs)) + cs
                    else:
                        s, e = float(s), float(e)
                    
                    if cs <= s < e <= ce: 
                        all_pairs.append((s, e))
            
            # Deduplica
            uniq = _dedup_segments(all_pairs)
            for (s, e) in uniq:
                d = e - s
                qs_rel = (s - cs) / clen
                qe_rel = (e - cs) / clen
                rows.append({
                    "d": d, 
                    "qs_rel": qs_rel, 
                    "qe_rel": qe_rel
                })
                
    df = pd.DataFrame(rows)
    if not df.empty:
        df["bucket"] = df["d"].apply(bucket_of)
    return df

def build_GEQ_from_generated(clips_data):
    """Costruisce il DataFrame GEQ (Generated Event Queries) dal JSON generato."""
    gen_rows = []
    for c in clips_data:
        cs = float(c["clip_start_sec"])
        ce = float(c["clip_end_sec"])
        clen = max(1e-9, ce - cs)
        
        for q in c.get("queries", []):
            s = float(q["query_start_sec"])
            e = float(q["query_end_sec"])
            d = e - s
            
            if clen <= 0: continue
            qs_rel = (s - cs)/clen
            qe_rel = (e - cs)/clen
            
            # Validazione base
            if not (0.0 <= qs_rel <= 1.0 and 0.0 <= qe_rel <= 1.0 and qs_rel < qe_rel):
                continue
            
            # Recupera bucket dal JSON o calcolalo
            bucket_raw = q.get("bucket", bucket_of(d))
            
            gen_rows.append({
                "d": d, 
                "qs_rel": qs_rel, 
                "qe_rel": qe_rel,
                "bucket": canonical_bucket(bucket_raw)
            })
    return pd.DataFrame(gen_rows)


def main():
    print("=== ANALISI COMPLETA CATALOGO SINTETICO ===")
    
    # --- 1. CARICAMENTO FILE ---
    print(f"\n[1/4] Lettura file...")
    
    # Load Train
    if not os.path.exists(TRAIN_JSON_PATH):
        print(f"ERRORE: File train non trovato: {TRAIN_JSON_PATH}")
        return
    with open(TRAIN_JSON_PATH, 'r') as f: 
        train_data = json.load(f)
    print(f" -> Train caricato.")

    # Load Generated
    if not os.path.exists(GENERATED_JSON_PATH):
        print(f"ERRORE: File generato non trovato: {GENERATED_JSON_PATH}")
        return
    with open(GENERATED_JSON_PATH, 'r') as f: 
        gen_json = json.load(f)
    
    if "clips" not in gen_json:
        print("ERRORE: JSON generato non valido (manca chiave 'clips').")
        return
    clips_list = gen_json["clips"]
    print(f" -> Generato caricato ({len(clips_list)} clips).")


    # --- 2. STATISTICHE GENERALI ---
    print(f"\n[2/4] Statistiche Descrittive Generato")
    num_clips = len(clips_list)
    num_queries = sum(len(c.get("queries", [])) for c in clips_list)
    avg_density = num_queries / num_clips if num_clips > 0 else 0

    print("="*40)
    print(f" Numero Totale Clip:    {num_clips}")
    print(f" Numero Totale Query:   {num_queries}")
    print(f" Densità Media:         {avg_density:.2f} query/clip")
    print("="*40)
    
    counts = [len(c.get("queries", [])) for c in clips_list]
    print(f"Distribuzione query per clip:")
    unique, counts_per_n = np.unique(counts, return_counts=True)
    for k, v in zip(unique, counts_per_n):
        print(f" - {k} query: {v} clip ({v/num_clips*100:.1f}%)")


    print(f"\n[3/4] Preparazione DataFrames (TRQ vs GEQ)...")
    TRQ = build_TRQ_from_train(train_data)
    GEQ = build_GEQ_from_generated(clips_list)
    
    if TRQ.empty or GEQ.empty:
        print("ERRORE: Uno dei dataframe è vuoto. Impossibile procedere con KS/Plot.")
        return
    
    print(f" -> TRQ (Train) rows: {len(TRQ)}")
    print(f" -> GEQ (Generated) rows: {len(GEQ)}")


    # --- 4. ANALISI KS & PLOTTING ---
    print(f"\n[4/4] Analisi Copertura Relativa (d_rel) per Bucket (KS Test + Plots)")
    print(f" -> I grafici verranno salvati in: {OUTPUT_IMG_DIR}")
    
    buckets_list = [(DURATION_BUCKETS[i], DURATION_BUCKETS[i+1]) for i in range(len(DURATION_BUCKETS)-1)]
    ks_results = []

    for b in buckets_list:
        # Filtra per bucket
        t_vals = TRQ[TRQ["bucket"] == b]
        g_vals = GEQ[GEQ["bucket"] == b]
        
        n_t = len(t_vals)
        n_g = len(g_vals)
        
        # Skip se dati insufficienti
        if n_t < 5 or n_g < 5:
            ks_results.append({"bucket": b, "KS": np.nan, "n_train": n_t, "n_gen": n_g})
            continue
            
        # Calcola Relative Coverage (d_rel)
        # d_rel = durata_query / durata_clip = qe_rel - qs_rel
        d_rel_train = t_vals["qe_rel"] - t_vals["qs_rel"]
        d_rel_gen   = g_vals["qe_rel"] - g_vals["qs_rel"]
        
        # KS Test
        ks_stat = ks_2samp(d_rel_train, d_rel_gen).statistic
        ks_results.append({"bucket": b, "KS": ks_stat, "n_train": n_t, "n_gen": n_g})

        # Plot ECDF
        plt.figure(figsize=(6, 4))
        xt, yt = ecdf(d_rel_train)
        xg, yg = ecdf(d_rel_gen)
        
        plt.plot(xt, yt, label=f"Train (n={n_t})", linewidth=2, color='tab:blue', alpha=0.8)
        plt.plot(xg, yg, label=f"Generated (n={n_g})", linewidth=2, linestyle="--", color='tab:orange', alpha=0.9)
        
        plt.title(f"ECDF Relative Coverage ($d_{{rel}}$)\nBucket {b}s - KS Stat: {ks_stat:.3f}")
        plt.xlabel("Relative Coverage (query len / clip len)")
        plt.ylabel("Cumulative Probability")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        
        # Salva
        filename = f"ecdf_drel_{int(b[0]):02d}_{int(b[1]):02d}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_IMG_DIR, filename), dpi=150)
        plt.close()

    # Stampa Tabella Finale KS
    print("\n=== RISULTATI TEST KS (Kolmogorov-Smirnov) ===")
    print("Nota: KS basso (<0.10/0.15) indica un buon matching della distribuzione.")
    df_ks = pd.DataFrame(ks_results)
    print(df_ks.to_string(index=False))
    print("\nAnalisi completata.")

if __name__ == "__main__":
    main()