import matplotlib.pyplot as plt
import statistics
from JobSetRandomGenerator import generate_jobs
from modello2.model2_tbpp_fu import solve_model2

def main():
    C = 100
    
    n_values = list(range(5, 18)) 
    num_seeds = 10  # Media su 10 istanze per pulire il grafico
    
    avg_runtimes = [] 
    
    print(f"=== INIZIO STUDIO DI SCALABILITÀ MODELLO 2 (Media su {num_seeds} istanze) ===")
    
    for n in n_values:
        print(f"Calcolo per n = {n:2d} |", end=" ", flush=True)
        
        runtimes_for_n = []
        
        for seed in range(42, 42 + num_seeds):
            # 1. Genera i job (che ora escono già come tuple!)
            jobs = generate_jobs(
                n=n,
                C=C,
                s_factor=1.0,
                duration_type="short",
                size_type="low",
                seed=seed
            )
                        
            # 2. Risolvi il Modello 2 in modo silenzioso passandogli direttamente "jobs"
            result = solve_model2(
                jobs=jobs,
                C=C,
                gamma=1.0,
                time_limit=1800,
                verbose=False, 
                binary_w=True
            )
            
            # 3. Salva il tempo
            if result["status"] in [2, 9]: # 2=OPTIMAL, 9=TIME_LIMIT
                tempo = result["runtime"]
            else:
                tempo = 0.0
                
            runtimes_for_n.append(tempo)
            
            # Puntino visivo
            print(".", end="", flush=True) 

        # 4. Calcola la media dei 10 tempi
        media_tempo = statistics.mean(runtimes_for_n)
        avg_runtimes.append(media_tempo)
        
        print(f"| Tempo Medio: {media_tempo:.4f} sec")

    print("\n=== STUDIO COMPLETATO. GENERAZIONE GRAFICO... ===")

    # ---------------------------------------------------------
    # Creazione del Grafico
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Plottiamo in Verde (color='g') per distinguerlo dal Modello 1
    plt.plot(n_values, avg_runtimes, marker='s', color='g', linestyle='-', linewidth=2, label="Model 2")
    
    plt.title(f'Scalabilità Modello 2: Tempo Medio di Risoluzione', fontsize=14)
    plt.xlabel('Numero di Job (n)', fontsize=12)
    plt.ylabel('Tempo Medio di Esecuzione (secondi)', fontsize=12)
    
    plt.xticks(n_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()