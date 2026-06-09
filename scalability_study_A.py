import os
import itertools
import statistics
import time
import matplotlib.pyplot as plt

# --- IMPORT MODELLI BASE ---
from JobSetRandomGenerator import generate_jobs
from modello1.model1_tbpp_fu import solve_model1
from modello2.model2_tbpp_fu import solve_model2
from modello3.model3_tbpp_fu import solve_model3

# --- IMPORT MODELLI OTTIMIZZATI (Futuri) ---
# Togli il commento a queste righe quando avrai creato i file dei modelli ottimizzati!
from modello1.model1_optimized_tbpp_fu import solve_model1_optimized
from modello2.model2_optimized_tbpp_fu import solve_model2_optimized
from modello3.model3_optimized_tbpp_fu import solve_model3_optimized

def esegui_suite(nome_suite, modelli, n_values, s_factors, durations, sizes, num_instances, time_limit, max_n_m1, C=100):
    """
    Esegue tutti i test per una specifica suite (Base o Ottimizzata),
    salva la tabella in un file TXT e genera il grafico PNG.
    """
    
    # NOTA: se cartella e file esistono già vengono sovrascritti silenziosamente senza crash

    # 1. Preparazione file di output
    os.makedirs("results_testbedA", exist_ok=True)
    file_tabella = f"results_testbedA/tab_{nome_suite}.txt"
    file_grafico = f"results_testbedA/grp_{nome_suite}.png"
    
    gruppi = list(itertools.product(n_values, s_factors, durations, sizes))
    
    # Dizionari per accumulare i tempi medî globali per il grafico (in base a n)
    tempi_medi_n = {n: {'m1': [], 'm2': [], 'm3': []} for n in n_values}

    # Apriamo il file di testo in modalità scrittura
    with open(file_tabella, "w", encoding="utf-8") as f:
        
        intestazione = f"\n=== SUITE: {nome_suite.upper()} (Gruppi: {len(gruppi)}, Seed/Grp: {num_instances}) ===\n"
        colonne = f"{'n':<4} | {'s_fact':<6} | {'Durat':<6} | {'Size':<4} || {'Mod 1 (s)':<12} | {'Mod 2 (s)':<12} | {'Mod 3 (s)':<12}\n"
        separatore = "-" * 85 + "\n"
        
        # Stampiamo su schermo E scriviamo su file
        print(intestazione + separatore + colonne + separatore, end="")
        f.write(intestazione + separatore + colonne + separatore)

        start_time = time.time()

        for (n, s_fact, dur, size) in gruppi:
            r1, r2, r3 = [], [], []
            
            for seed in range(42, 42 + num_instances):
                jobs = generate_jobs(n, C, s_fact, dur, size, seed)
                
                # --- MODELLO 1 ---
                if n <= max_n_m1:
                    if nome_suite == "Base":
                        res1 = modelli['m1'](jobs=jobs, C=C, gamma=1.0, time_limit=time_limit, verbose=False, binary_w=True)
                    else:
                        res1 = modelli['m1'](jobs=jobs, C=C, gamma=1.0, time_limit=time_limit, verbose=False)
                    
                    t1 = res1.get("runtime", time_limit) if res1["status"] in [2, 9] else time_limit
                    r1.append(t1)
                else:
                    r1.append(None) # OOT
                    
                # --- MODELLO 2 ---
                if nome_suite == "Base":
                    res2 = modelli['m2'](jobs=jobs, C=C, gamma=1.0, time_limit=time_limit, verbose=False, binary_w=True)
                else:
                    res2 = modelli['m2'](jobs=jobs, C=C, gamma=1.0, time_limit=time_limit, verbose=False)
                
                t2 = res2.get("runtime", time_limit) if res2["status"] in [2, 9] else time_limit
                r2.append(t2)
                
                # --- MODELLO 3 ---
                # Nessuno dei due Modelli 3 usa binary_w
                res3 = modelli['m3'](jobs=jobs, C=C, gamma=1.0, time_limit=time_limit, verbose=False)
                t3 = res3.get("runtime", time_limit) if res3["status"] in [2, 9] else time_limit
                r3.append(t3)

            # Calcolo Medie per il gruppo (per la riga della tabella)
            avg_m2 = statistics.mean(r2)
            avg_m3 = statistics.mean(r3)
            
            if None in r1:
                str_m1 = "OOT/Skip"
                avg_m1 = None
            else:
                avg_m1 = statistics.mean(r1)
                str_m1 = f"{avg_m1:.3f}"
                
            riga = f"{n:<4} | {s_fact:<6} | {dur:<6} | {size:<4} || {str_m1:<12} | {avg_m2:<12.3f} | {avg_m3:<12.3f}\n"
            print(riga, end="")
            f.write(riga)
            
            # Salvataggio dati aggregati per il grafico
            if avg_m1 is not None:
                tempi_medi_n[n]['m1'].append(avg_m1)
            tempi_medi_n[n]['m2'].append(avg_m2)
            tempi_medi_n[n]['m3'].append(avg_m3)

        tempo_tot = (time.time() - start_time) / 60
        chiusura = separatore + f"TEST COMPLETATO IN {tempo_tot:.1f} MINUTI.\n"
        print(chiusura)
        f.write(chiusura)

    # ---------------------------------------------------------
    # 2. Generazione e Salvataggio del Grafico (Nessun blocco!)
    # ---------------------------------------------------------
    asse_x = []
    y_m1, y_m2, y_m3 = [], [], []

    for n in n_values:
        asse_x.append(n)
        # Calcoliamo la media globale su tutti i gruppi per il grafico
        y_m1.append(statistics.mean(tempi_medi_n[n]['m1']) if tempi_medi_n[n]['m1'] else None)
        y_m2.append(statistics.mean(tempi_medi_n[n]['m2']))
        y_m3.append(statistics.mean(tempi_medi_n[n]['m3']))

    plt.figure(figsize=(10, 6))
    
    # Plottiamo M1 solo per i punti validi (non None)
    valid_m1_x = [x for x, y in zip(asse_x, y_m1) if y is not None]
    valid_m1_y = [y for y in y_m1 if y is not None]
    
    if valid_m1_x:
        plt.plot(valid_m1_x, valid_m1_y, marker='o', color='b', linestyle='-', linewidth=2, label="Model 1")
        
    plt.plot(asse_x, y_m2, marker='s', color='g', linestyle='--', linewidth=2, label="Model 2")
    plt.plot(asse_x, y_m3, marker='^', color='r', linestyle='-.', linewidth=2, label="Model 3")
    
    plt.title(f'Testbed A: Scalabilità Modelli {nome_suite.capitalize()}', fontsize=14, fontweight='bold')
    plt.xlabel('Numero di Job (n)', fontsize=12)
    plt.ylabel('Tempo Medio (secondi)', fontsize=12)
    plt.xticks(asse_x)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    # SALVATAGGIO INVECE DI SHOW
    plt.savefig(file_grafico, bbox_inches='tight')
    plt.close() # Libera la memoria
    
    print(f">> Dati salvati in: {file_tabella}")
    print(f">> Grafico salvato in: {file_grafico}\n")


def main():
    # --- CONFIGURAZIONE LEVE ---
    # =====================================================================
    # LEVE DI COMPUTAZIONE (MODIFICA QUESTI VALORI PER VELOCIZZARE IL TEST)
    # =====================================================================
    
    # 1. Dimensioni del problema. 
    # Paper originale: [50, 100, 150, 200]
    # Se vuoi testare anche il Modello 1, usa: [10, 15, 20, 25]
    #n_values = [15, 20, 50, 100] 
    n_values = [5, 10, 15, 20]
    
    # 2. Numero di istanze per gruppo (seed). 
    # Paper originale: 5. 
    # Metti 2 o 3 per fare un test rapido.
    NUM_INSTANCES = 2  
    
    # 3. Tempo massimo per Gurobi (in secondi). 
    # Paper originale: 1800 (30 minuti). 
    # Metti 60 o 120 per testare se Gurobi si blocca prima.
    TIME_LIMIT = 60 
    
    # 4. SALVAVITA PER IL MODELLO 1:
    # Oltre questo valore di 'n', lo script non avvierà nemmeno il Modello 1 
    # per evitare che il PC vada in crash (Out Of Memory).
    MAX_N_MODEL1 = 20
    
    # =====================================================================

    # Altri indicatori del Testbed A
    s_factors = [1.0, 1.2]
    durations = ["short", "long"]
    sizes = ["low", "high"]
    
    # ---------------------------------------------------------
    # SUITE 1: MODELLI BASE
    # ---------------------------------------------------------
    modelli_base = {
        'm1': solve_model1,
        'm2': solve_model2,
        'm3': solve_model3
    }
    
    esegui_suite("Base", modelli_base, n_values, s_factors, durations, sizes, NUM_INSTANCES, TIME_LIMIT, MAX_N_MODEL1)
    
    # ---------------------------------------------------------
    # SUITE 2: MODELLI OTTIMIZZATI
    # ---------------------------------------------------------
    
    modelli_ottimizzati = {
        'm1': solve_model1_optimized,
        'm2': solve_model2_optimized,
        'm3': solve_model3_optimized
    }
    
    esegui_suite("Ottimizzati", modelli_ottimizzati, n_values, s_factors, durations, sizes, NUM_INSTANCES, TIME_LIMIT, MAX_N_MODEL1)
    
    print("=== TUTTE LE SUITE SONO STATE COMPLETATE CON SUCCESSO! ===")

if __name__ == "__main__":
    main()