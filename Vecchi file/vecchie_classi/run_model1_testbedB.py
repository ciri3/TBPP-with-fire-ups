from TestbedBGenerator import generate_testbed_B
from modello1.model1_tbpp_fu import solve_model1

def main():
    C = 100

    print("=== Generazione dei job (Testbed B) ===")

    # Usiamo il nuovo generatore invece di generate_jobs!
    # Esempio: 15 istanti di tempo, da 5 a 10 job attivi simultaneamente,
    # con alta sopravvivenza (b=70-90%)
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

    print("\n=== Risoluzione Modello 1 ===")
    
    # La chiamata al modello rimane ASSOLUTAMENTE IDENTICA!
    result = solve_model1(
        jobs=jobs,
        C=C,
        gamma=1.0,
        time_limit=1800,
        verbose=True,  # Mettiamo True per vedere Gurobi che lavora
        binary_w=True
    )

    print("\nRisultato modello:")
    print(f"Status: {result['status']}")
    print(f"Objective: {result.get('objective', 'N/A')}")
    print(f"Runtime: {result.get('runtime', 0.0):.4f} secondi")


if __name__ == "__main__":
    main()