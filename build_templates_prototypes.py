import os
import json
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


BASE_PATH = "" 
TRAIN_JSON_PATH = os.path.join(BASE_PATH, "nlq_train.json")
NARR_JSON_SAN   = os.path.join(BASE_PATH, "narration.json")

# Salveremo due file: uno per i vettori (.npy) e uno per i nomi (.json)
OUTPUT_NPY      = "template_prototypes.npy"
OUTPUT_META     = "template_metadata.json"

MODEL_NAME      = "sentence_transformers/all-MiniLM-L6-v2"


tqdm.pandas()

def main():
    print("--- INIZIO CREAZIONE PROTOTIPI---")
    # FASE 1: Caricamento Training Set
    print(f"[1/5] Caricamento NLQ Train: {TRAIN_JSON_PATH}")
    if not os.path.exists(TRAIN_JSON_PATH):
        raise FileNotFoundError(f"File non trovato: {TRAIN_JSON_PATH}")
        
    with open(TRAIN_JSON_PATH, 'r') as f:
        train_data = json.load(f)

    records = []
    for video in train_data['videos']:
        video_uid = video['video_uid']
        for clip in video['clips']:
            for annotation in clip['annotations']:
                for query in annotation['language_queries']:
                    if query.get('template'):
                        records.append({
                            'video_uid': video_uid,
                            'template': query['template'],
                            'start_sec': query['clip_start_sec'],
                            'end_sec': query['clip_end_sec']
                        })
    df_queries = pd.DataFrame(records)
    print(f"   -> Trovate {len(df_queries)} query con template.")

    # FASE 2: Caricamento Narrazioni
    print(f"[2/5] Caricamento Narrazioni: {NARR_JSON_SAN}")
    if not os.path.exists(NARR_JSON_SAN):
        raise FileNotFoundError(f"File non trovato: {NARR_JSON_SAN}")

    with open(NARR_JSON_SAN, 'r') as f:
        narrations_data = json.load(f)

    def get_narrations_for_query(video_uid, start_sec, end_sec, narrations_dict):
        narrations_text = []
        if video_uid in narrations_dict:
            pass_data = narrations_dict[video_uid].get("narration_pass_1", {})
            narration_list = pass_data.get("narrations", [])
            for narration in narration_list:
                if not narration.get('timestamp_sec') or not narration.get('narration_text'):
                    continue
                narration_time = float(narration['timestamp_sec'])
                if start_sec <= narration_time <= end_sec:
                    narrations_text.append(narration['narration_text'])
        return " ".join(narrations_text)

    # FASE 3: Associazione Query -> Narrazioni
    print("[3/5] Associazione Narrazioni alle Query...")
    df_queries['narrations'] = df_queries.progress_apply(
        lambda row: get_narrations_for_query(
            row['video_uid'],
            row['start_sec'],
            row['end_sec'],
            narrations_data
        ),
        axis=1
    )
    
    df_clean = df_queries[df_queries['narrations'].str.strip() != ''].copy()
    print(f"   -> Query utili: {len(df_clean)}")

    # FASE 4: Calcolo Embeddings e Prototipi
    print(f"[4/5] Calcolo Prototipi con modello: {MODEL_NAME}")
    
    df_templates = (
        df_clean
        .groupby('template')['narrations']
        .apply(lambda x: ' '.join(x))
        .reset_index()
        .rename(columns={'narrations': 'narrations_text'})
    )

    model = SentenceTransformer(MODEL_NAME)
    SENT_SPLIT = re.compile(r'[.!?]\s+')

    def split_sentences(text):
        parts = [p.strip() for p in SENT_SPLIT.split(text) if p.strip()]
        return parts if parts else [text.strip()]

    template_names = df_templates['template'].tolist()
    template_to_sentences = {}
    
    for _, row in df_templates.iterrows():
        t = row['template']
        sentences = split_sentences(row['narrations_text'])
        sentences = [s for s in sentences if len(s) > 2]
        template_to_sentences[t] = sentences

    proto_vectors = []
    valid_template_names = []

    print("-> Generazione Embeddings...")
    for t in tqdm(template_names, desc="Encoding Templates"):
        sents = template_to_sentences.get(t, [])
        if not sents:
            continue
            
        emb = model.encode(sents, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        proto = emb.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12) # Normalizzazione L2
        
        proto_vectors.append(proto.astype(np.float32))
        valid_template_names.append(t)

    prototypes_matrix = np.vstack(proto_vectors).astype(np.float32)
    print(f"   -> Matrice creata: {prototypes_matrix.shape}")

    # FASE 5: Salvataggio .NPY + .JSON
    print(f"[5/5] Salvataggio...")
    
    # 1. Salva la matrice numerica in .npy
    np.save(OUTPUT_NPY, prototypes_matrix)
    print(f"   -> Matrice salvata in: {OUTPUT_NPY}")

    # 2. Salva i metadati (nomi dei template) in .json
    metadata = {
        "template_names": valid_template_names,
        "model_name": MODEL_NAME
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   -> Metadati salvati in: {OUTPUT_META}")

    print("\n Operazione Completata")

if __name__ == "__main__":
    main()