from JobSetRandomGenerator import generate_jobs
from modello1.model1_tbpp_fu import solve_model1


def main():
    C = 100

    print("Generazione dei job...")

    jobs = generate_jobs(
        n=17,
        C=C,
        s_factor=1.0,
        duration_type="short",
        size_type="low",
        seed=42
    )

    print("Job generati:")
    for i, job in enumerate(jobs, start=1):
        print(f"Job {i}: s={job[0]}, e={job[1]}, c={job[2]}")

    #print("Job generati:")
    #for i, job in enumerate(jobs, start=1):
    #    print(f"Job {i}: {job}")


    result = solve_model1(
        jobs=jobs,
        C=C,
        gamma=1.0,
        time_limit=1800,
        verbose=True,
        binary_w=True
    )

    print("\nRisultato modello:")
    print(result)


if __name__ == "__main__":
    main()