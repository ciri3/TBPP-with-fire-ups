import matplotlib.pyplot as plt
import statistics

# Importiamo il generatore e i tre modelli
from TestbedBGenerator import generate_testbed_B
from modello1.model1_tbpp_fu import solve_model1
from modello2.model2_tbpp_fu import solve_model2
from modello3.model3_tbpp_fu import solve_model3

def main():
    C = 100
    
    # Facciamo variare l'orizzonte temporale |T|
    # NOTA: Se a 25 ci mette troppo tempo, abbassa a 20! (ctrl/cmd+C per interrompere)
    t_values = [5, 10, 15, 20, 25] 
    num_seeds = 5  # Media su 5 istanze per velocità
    
    # Parametri fissi per il Testbed B
    A_MIN, A_MAX = 5, 10
    B_MIN, B_MAX = 60, 80
    
    # Liste per i tempi medi dei tre modelli
    avg_runtimes_m1 = []
    avg_runtimes_m2 = []
    avg_runtimes_m3 = []
    
    print("=== CONFRONTO GLOBALE MODELLI - TESTBED B ===")
    print(f"Parametri: a=[{A_MIN},{A_MAX}], b=[{B_MIN}%,{B_MAX}%], Seeds={num_seeds}")
    print("-" * 75)
    print(f"{'|T|':<4} | {'Avg Job (n)':<11} | {'Mod 1 (s)':<14} | {'Mod 2 (s)':<14} | {'Mod 3 (s)':<14}")
    print("-" * 75)
    
    for T_size in t_values:
        runtimes_m1_n = []
        runtimes_m2_n = []
        runtimes_m3_n = []
        n_totali = []
        
        for seed in range(42, 42 + num_seeds):
            # 1. Genera l'istanza comune
            jobs = generate_testbed_B(T_size, A_MIN, A_MAX, B_MIN, B_MAX, seed)
            n_totali.append(len(jobs))
            
            # --- TEST MODELLO 1 ---
            res1 = solve_model1(jobs=jobs, C=C, gamma=1.0, time_limit=1800, verbose=False, binary_w=True)
            t1 = res1.get("runtime", 0.0) if res1["status"] in [2, 9] else 0.0
            runtimes_m1_n.append(t1)
            
            # --- TEST MODELLO 2 ---
            res2 = solve_model2(jobs=jobs, C=C, gamma=1.0, time_limit=1800, verbose=False, binary_w=True)
            t2 = res2.get("runtime", 0.0) if res2["status"] in [2, 9] else 0.0
            runtimes_m2_n.append(t2)
            
            # --- TEST MODELLO 3 ---
            res3 = solve_model3(jobs=jobs, C=C, gamma=1.0, time_limit=1800, verbose=False)
            t3 = res3.get("runtime", 0.0) if res3["status"] in [2, 9] else 0.0
            runtimes_m3_n.append(t3)

        # 2. Calcolo delle medie
        media_n = statistics.mean(n_totali)
        media_m1 = statistics.mean(runtimes_m1_n)
        media_m2 = statistics.mean(runtimes_m2_n)
        media_m3 = statistics.mean(runtimes_m3_n)
        
        # 3. Salvataggio per il grafico
        avg_runtimes_m1.append(media_m1)
        avg_runtimes_m2.append(media_m2)
        avg_runtimes_m3.append(media_m3)
        
        # 4. Log in tempo reale sulla riga della tabella
        print(f"{T_size:<4} | {media_n:<11.1f} | {media_m1:<14.4f} | {media_m2:<14.4f} | {media_m3:<14.4f}")

    print("-" * 75)
    print("=== TEST COMPLETATI. GENERAZIONE GRAFICO... ===")

    # ---------------------------------------------------------
    # Creazione del Grafico Comparativo
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 7))
    
    plt.plot(t_values, avg_runtimes_m1, marker='o', color='b', linestyle='-', linewidth=2, label="Model 1 (No Preprocessing)")
    plt.plot(t_values, avg_runtimes_m2, marker='s', color='g', linestyle='--', linewidth=2, label="Model 2 (Partial Preprocessing)")
    plt.plot(t_values, avg_runtimes_m3, marker='^', color='r', linestyle='-.', linewidth=2, label="Model 3 (Job-to-Job Assignment)")
    
    plt.title('TBPP-FU Testbed B: Confronto di Scalabilità sull\'Orizzonte Temporale', fontsize=16, fontweight='bold')
    plt.xlabel('Orizzonte Temporale (|T|)', fontsize=14)
    plt.ylabel('Tempo Medio di Esecuzione (secondi)', fontsize=14)
    
    plt.xticks(t_values)
    plt.grid(True, linestyle=':', alpha=0.7, color='gray')
    plt.legend(loc='upper left', fontsize=12)
    plt.gca().set_facecolor('#f9f9f9')
    
    plt.show()

if __name__ == "__main__":
    main()