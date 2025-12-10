import sqlite3
import os
import time

# ---------- CONFIGURA IL PERCORSO ----------
DB_PATH = "pretrain_catalog/catalog_v2.db" 
# -------------------------------------------

def add_density_score():
    if not os.path.exists(DB_PATH):
        print(f"ERRORE: Il file {DB_PATH} non esiste.")
        return

    print(f"Connessione al database: {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # 1. Creazione Indice (FONDAMENTALE per la velocità)
        print("[1/3] Creazione indice per ottimizzare il calcolo...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_density_calc ON candidates(video_uid, cs, ce, _len)")
        conn.commit()

        # 2. Aggiunta Colonna (se non esiste)
        print("[2/3] Aggiunta colonna 'density_score'...")
        try:
            cursor.execute("ALTER TABLE candidates ADD COLUMN density_score INTEGER DEFAULT 0")
            conn.commit()
        except sqlite3.OperationalError:
            print("      (La colonna esiste già, procedo all'aggiornamento)")

        # 3. Calcolo Densità (La query "Sonar")
        print("[3/3] Calcolo densità in corso (potrebbe richiedere qualche secondo/minuto)...")
        start_time = time.time()
        
        # La logica: Per ogni candidato "lungo" (>10s), conta quanti candidati "corti" (<10s)
        # contiene interamente o quasi.
        sql_update = """
        UPDATE candidates 
        SET density_score = (
            SELECT COUNT(*)
            FROM candidates AS neighbors
            WHERE neighbors.video_uid = candidates.video_uid
              AND neighbors.id != candidates.id
              AND neighbors._len < 10.0       -- Contiamo solo le query brevi (i "pesci")
              AND neighbors.cs >= candidates.cs 
              AND neighbors.ce <= candidates.ce
        )
        WHERE _len >= 10.0; -- Calcoliamo lo score solo per i potenziali "contenitori"
        """
        
        cursor.execute(sql_update)
        rows_affected = cursor.rowcount
        conn.commit()
        
        elapsed = time.time() - start_time
        print(f"\n[SUCCESSO] Aggiornati {rows_affected} candidati (container) in {elapsed:.2f} secondi.")
        
        # Verifica rapida
        cursor.execute("SELECT MAX(density_score) FROM candidates")
        max_score = cursor.fetchone()[0]
        print(f"Max Density Score trovato nel DB: {max_score}")
        
        if max_score == 0:
             print("[WARN] Attenzione: Il max score è 0. Significa che nessun container ha trovato vicini.")
             print("       Controlla che 'make_catalog_short' abbia salvato candidati < 10s.")

    except Exception as e:
        print(f"\n[ERRORE] Qualcosa è andato storto: {e}")
        conn.rollback()
    finally:
        conn.close()
        print("Connessione chiusa.")

if __name__ == "__main__":
    add_density_score()