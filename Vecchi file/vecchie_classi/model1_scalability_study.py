import matplotlib.pyplot as plt
import statistics
from JobSetRandomGenerator import generate_jobs
from modello1.model1_tbpp_fu import solve_model1

def main():
    C = 100
    
    # I nostri parametri
    n_values = list(range(5, 18)) # non ha senso iniziare prima di 5 (da modificare con licenza Gurobi per oltre 17)
    num_seeds = 10  # Statistica: quante istanze diverse risolvere per ogni n
    
    avg_runtimes = [] # Qui salveremo i tempi MEDI
    
    print(f"=== INIZIO STUDIO DI SCALABILITÀ (Media su {num_seeds} istanze) ===")
    
    for n in n_values:
        print(f"Calcolo per n = {n:2d} |", end=" ", flush=True)
        
        runtimes_for_n = [] # Tempi per i 10 seed di questo specifico n
        
        # Facciamo un ciclo sui seed (es. da 42 a 51)
        for seed in range(42, 42 + num_seeds):
            # 1. Genera i job con il seed corrente
            jobs = generate_jobs(
                n=n,
                C=C,
                s_factor=1.0,
                duration_type="short",
                size_type="low",
                seed=seed
            )
            
            # 2. Risolvi il modello in modo silenzioso
            result = solve_model1(
                jobs=jobs,
                C=C,
                gamma=1.0,
                time_limit=1800,
                verbose=False, 
                binary_w=True
            )
            
            # 3. Salva il tempo
            if result["status"] in [2, 9]: # 2 = Gurobi OPTIMAL, 9 = massimo limite di tempo raggiunto
                tempo = result["runtime"]
            else: #es: PB non risolvibile o tendente a infinito
                tempo = 0.0 # evita di rovinare le statistiche con tempi di 0,001s che generano l'errore
                
            runtimes_for_n.append(tempo)
            
            # Stampa un puntino per dare un feedback visivo che sta lavorando
            print(".", end="", flush=True) 

        # 4. Calcola la media dei 10 tempi
        media_tempo = statistics.mean(runtimes_for_n)
        avg_runtimes.append(media_tempo)
        
        # Mostra la media finale per questo n
        print(f"| Tempo Medio: {media_tempo:.4f} sec")

    print("\n=== STUDIO COMPLETATO. GENERAZIONE GRAFICO... ===")

    # ---------------------------------------------------------
    # 5. Creazione del Grafico
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Adesso plottiamo n_values contro le MEDIE
    plt.plot(n_values, avg_runtimes, marker='o', color='b', linestyle='-', linewidth=2)
    
    plt.title(f'Scalabilità Modello 1: Tempo Medio su {num_seeds} Istanze', fontsize=14)
    plt.xlabel('Numero di Job (n)', fontsize=12)
    plt.ylabel('Tempo Medio di Esecuzione (secondi)', fontsize=12)
    
    plt.xticks(n_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()

if __name__ == "__main__":
    main()