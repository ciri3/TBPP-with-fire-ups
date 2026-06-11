import matplotlib.pyplot as plt
import statistics
from JobSetRandomGenerator import generate_jobs

# Assicurati che il percorso di importazione sia corretto
from modello3.model3_tbpp_fu import solve_model3

def main():
    C = 100
    
    # Spingiamoci fino a n=30 per vedere quanto è potente questo Modello 3!
    n_values = list(range(5, 26)) 
    num_seeds = 10 
    
    avg_runtimes = [] 
    
    print(f"=== INIZIO STUDIO DI SCALABILITÀ MODELLO 3 (Media su {num_seeds} istanze) ===")
    
    for n in n_values:
        print(f"Calcolo per n = {n:2d} |", end=" ", flush=True)
        
        runtimes_for_n = []
        
        for seed in range(42, 42 + num_seeds):
            # Genera i job (escono già in formato tupla)
            jobs = generate_jobs(
                n=n,
                C=C,
                s_factor=1.0,
                duration_type="short",
                size_type="low",
                seed=seed
            )
            
            # Risolvi il Modello 3 (nota: non c'è il parametro binary_w qui, come dal tuo file)
            result = solve_model3(
                jobs=jobs,
                C=C,
                gamma=1.0,
                time_limit=1800,
                verbose=False
            )
            
            # Salva il tempo
            if result["status"] in [2, 9]: # 2=OPTIMAL, 9=TIME_LIMIT
                tempo = result["runtime"]
            else:
                tempo = 0.0
                
            runtimes_for_n.append(tempo)
            
            print(".", end="", flush=True) 

        # Calcola la media dei 10 tempi
        media_tempo = statistics.mean(runtimes_for_n)
        avg_runtimes.append(media_tempo)
        
        print(f"| Tempo Medio: {media_tempo:.4f} sec")

    print("\n=== STUDIO COMPLETATO. GENERAZIONE GRAFICO... ===")

    # ---------------------------------------------------------
    # Creazione del Grafico
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Plottiamo in Rosso (color='r') e usiamo dei triangolini (marker='^')
    plt.plot(n_values, avg_runtimes, marker='^', color='r', linestyle='-', linewidth=2, label="Model 3")
    
    plt.title('Scalabilità Modello 3: Tempo Medio di Risoluzione', fontsize=14)
    plt.xlabel('Numero di Job (n)', fontsize=12)
    plt.ylabel('Tempo Medio di Esecuzione (secondi)', fontsize=12)
    
    # Mettiamo i tick sull'asse X ogni 2 numeri per non sovrapporre le scritte
    plt.xticks(range(5, 31, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()