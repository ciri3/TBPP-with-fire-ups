import matplotlib.pyplot as plt
import statistics
from TestbedBGenerator import generate_testbed_B
from modello2.model2_tbpp_fu import solve_model2

def main():
    C = 100
    t_values = [5, 10, 15, 20, 25]
    num_seeds = 5
    
    A_MIN, A_MAX = 5, 10
    B_MIN, B_MAX = 60, 80
    
    avg_runtimes = []
    
    print(f"=== SCALABILITÀ MODELLO 2 - TESTBED B (Media su {num_seeds} seed) ===")
    
    for T_size in t_values:
        print(f"Calcolo per |T| = {T_size:2d} |", end=" ", flush=True)
        runtimes = []
        n_totali = []
        
        for seed in range(42, 42 + num_seeds):
            jobs = generate_testbed_B(T_size, A_MIN, A_MAX, B_MIN, B_MAX, seed)
            n_totali.append(len(jobs))
            
            result = solve_model2(
                jobs=jobs, C=C, gamma=1.0, time_limit=1800, verbose=False, binary_w=True
            )
            
            tempo = result["runtime"] if result["status"] in [2, 9] else 0.0
            runtimes.append(tempo)
            print(".", end="", flush=True)
            
        media_tempo = statistics.mean(runtimes)
        media_n = statistics.mean(n_totali)
        avg_runtimes.append(media_tempo)
        print(f"| Job medi: {media_n:.1f} | Tempo: {media_tempo:.4f} sec")

    # Grafico
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, avg_runtimes, marker='s', color='g', linestyle='-', linewidth=2, label="Model 2")
    plt.title('Scalabilità Modello 2 (Testbed B)', fontsize=14)
    plt.xlabel('Orizzonte Temporale (|T|)', fontsize=12)
    plt.ylabel('Tempo Medio (secondi)', fontsize=12)
    plt.xticks(t_values)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()