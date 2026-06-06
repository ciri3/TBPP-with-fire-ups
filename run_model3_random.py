from JobSetRandomGenerator import generate_jobs
from modello3.model3_tbpp_fu import solve_model3


def main():
    C = 100

    print("Generazione dei job...")

    jobs = generate_jobs(
        n=20,
        C=C,
        s_factor=1.0,
        duration_type="short",
        size_type="low",
        seed=42
    )

    print("Job generati:")
    for i, job in enumerate(jobs, start=1):
        s, e, c = job
        print(f"Job {i}: s={s}, e={e}, c={c}")

    print("\nRisoluzione Modello 3...")

    result = solve_model3(
        jobs=jobs,
        C=C,
        gamma=1.0,
        time_limit=1800,
        verbose=True
    )

    print("\n--- RISULTATO MODELLO 3 ---")
    print("Status:", result["status"])
    print("Objective:", result["objective"])
    print("Server usati:", result["servers_used"])
    print("Fire-up:", result["fireups"])

    print("\nAssegnamento:")
    for initializer, assigned_jobs in result["assignment"].items():
        print(f"Server inizializzato dal job originale {initializer + 1}:")
        for job_id in assigned_jobs:
            print(f"  - Job originale {job_id + 1}")


if __name__ == "__main__":
    main()