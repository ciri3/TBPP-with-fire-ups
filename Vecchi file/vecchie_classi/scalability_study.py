import matplotlib.pyplot as plt
import statistics
from JobSetRandomGenerator import generate_jobs

from modello1.model1_tbpp_fu import solve_model1
from modello2.model2_tbpp_fu import solve_model2
from modello3.model3_tbpp_fu import solve_model3

def main():
    C = 100
    
    n_values = list(range(5, 19)) # da aumentare con licenza Gurobi
    num_seeds = 5  # Media su 5 istanze per velocizzare il confronto globale
    
    # Liste per salvare i tempi medi dei tre modelli
    avg_runtimes_m1 = []
    avg_runtimes_m2 = []
    avg_runtimes_m3 = []
    
    print(f"=== INIZIO CONFRONTO GLOBALE MODELLI (Media su {num_seeds} istanze) ===")
    print("-------------------------------------------------------------------")
    print(f"{'n':<4} | {'Model 1 (sec)':<15} | {'Model 2 (sec)':<15} | {'Model 3 (sec)':<15}")
    print("-------------------------------------------------------------------")
    
    for n in n_values:
        runtimes_m1_n = []
        runtimes_m2_n = []
        runtimes_m3_n = []
        
        for seed in range(42, 42 + num_seeds):
            # 1. Genera l'istanza comune per questo seed
            jobs = generate_jobs(
                n=n,
                C=C,
                s_factor=1.0,
                duration_type="short",
                size_type="low",
                seed=seed
            )
            
            # --- TEST MODELLO 1 ---
            res1 = solve_model1(jobs=jobs, C=C, gamma=1.0, time_limit=1800, verbose=False, binary_w=True)
            t1 = res1["runtime"] if res1["status"] in [2, 9] else 0.0
            runtimes_m1_n.append(t1)
            
            # --- TEST MODELLO 2 ---
            res2 = solve_model2(jobs=jobs, C=C, gamma=1.0, time_limit=1800, verbose=False, binary_w=True)
            t2 = res2["runtime"] if res2["status"] in [2, 9] else 0.0
            runtimes_m2_n.append(t2)
            
            # --- TEST MODELLO 3 ---
            res3 = solve_model3(jobs=jobs, C=C, gamma=1.0, time_limit=1800, verbose=False)
            t3 = res3["runtime"] if res3["status"] in [2, 9] else 0.0
            runtimes_m3_n.append(t3)

        # 2. Calcola le medie
        media_m1 = statistics.mean(runtimes_m1_n)
        media_m2 = statistics.mean(runtimes_m2_n)
        media_m3 = statistics.mean(runtimes_m3_n)
        
        # 3. Salva nelle liste per il grafico
        avg_runtimes_m1.append(media_m1)
        avg_runtimes_m2.append(media_m2)
        avg_runtimes_m3.append(media_m3)
        
        # 4. Stampa una riga di log formattata per vedere i progressi dal vivo
        print(f"{n:<4} | {media_m1:<15.4f} | {media_m2:<15.4f} | {media_m3:<15.4f}")

    print("-------------------------------------------------------------------")
    print("=== TEST COMPLETATI. GENERAZIONE GRAFICO... ===")

    # ---------------------------------------------------------
    # Creazione del Grafico Comparativo
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 7))
    
    # Disegna le tre linee con colori e stili diversi
    plt.plot(n_values, avg_runtimes_m1, marker='o', color='b', linestyle='-', linewidth=2, label="Model 1 (No Preprocessing)")
    plt.plot(n_values, avg_runtimes_m2, marker='s', color='g', linestyle='--', linewidth=2, label="Model 2 (Partial Preprocessing)")
    plt.plot(n_values, avg_runtimes_m3, marker='^', color='r', linestyle='-.', linewidth=2, label="Model 3 (Job-to-Job Assignment)")
    
    # Titoli e formattazione
    plt.title('TBPP-FU: Studio di Scalabilità e Confronto Modelli', fontsize=16, fontweight='bold')
    plt.xlabel('Numero di Job (n)', fontsize=14)
    plt.ylabel('Tempo Medio di Esecuzione (secondi)', fontsize=14)
    
    # Migliora la leggibilità dell'asse X e della griglia
    plt.xticks(n_values)
    plt.grid(True, linestyle=':', alpha=0.7, color='gray')
    
    # Mostra la legenda nell'angolo in alto a sinistra
    plt.legend(loc='upper left', fontsize=12)
    
    # Sfondo leggermente grigio chiaro per stile accademico (opzionale)
    plt.gca().set_facecolor('#f9f9f9')
    
    # Mostra il grafico
    plt.show()

if __name__ == "__main__":
    main()