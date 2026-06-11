from TestbedBGenerator import generate_testbed_B
from modello3.model3_tbpp_fu import solve_model3

def main():
    C = 100

    print("=== Generazione dei job (Testbed B) ===")

    # Parametri di test identici ai precedenti per coerenza
    jobs = generate_testbed_B(
        T_size=15, 
        a_min=5, 
        a_max=10, 
        b_min=70, 
        b_max=90, 
        seed=42
    )

    print(f"Job generati (Totale: {len(jobs)}):")
    for i, job in enumerate(jobs, start=1):
        print(f"Job {i:2d}: s={job[0]:2d}, e={job[1]:2d}, c={job[2]:3d}")

    print("\n=== Risoluzione Modello 3 ===")
    
    # Chiamata al Modello 3 (nota: senza binary_w=True)
    result = solve_model3(
        jobs=jobs,
        C=C,
        gamma=1.0,
        time_limit=1800,
        verbose=True
    )

    print("\nRisultato modello:")
    print(f"Status: {result['status']}")
    print(f"Objective: {result.get('objective', 'N/A')}")
    print(f"Runtime: {result.get('runtime', 0.0):.4f} secondi")

if __name__ == "__main__":
    main()