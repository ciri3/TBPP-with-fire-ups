# compare_model1.py
# fatto da ciri
from JobSetRandomGenerator import generate_jobs
from modello1.model1_tbpp_fu import solve_model1
from modello1.model1_optimized_tbpp_fu import solve_model1_optimized


def compare_once(
    n=20,
    C=100,
    gamma=1.0,
    seed=42,
    s_factor=1.0,
    duration_type="short",
    size_type="low",
    time_limit=1800
):
    jobs = generate_jobs(
        n=n,
        C=C,
        s_factor=s_factor,
        duration_type=duration_type,
        size_type=size_type,
        seed=seed
    )

    print("\n========== ISTANZA ==========")
    for i, (s, e, c) in enumerate(jobs):
        print(f"Job {i}: s={s}, e={e}, c={c}")

    print("\n========== MODELLO 1 BASE ==========")
    res_base = solve_model1(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=time_limit,
        verbose=True,
        binary_w=True
    )

    print("\n========== MODELLO 1 OTTIMIZZATO ==========")
    res_opt = solve_model1_optimized(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=time_limit,
        verbose=True
    )

    print("\n========== CONFRONTO ==========")
    print(f"Objective base:        {res_base['objective']}")
    print(f"Objective ottimizzato: {res_opt['objective']}")

    print(f"Server base:           {res_base['servers_used']}")
    print(f"Server ottimizzato:    {res_opt['servers_used']}")

    print(f"Fire-up base:          {res_base['fireups']}")
    print(f"Fire-up ottimizzato:   {res_opt['fireups']}")

    print(f"Runtime base:          {res_base['runtime']:.4f} s")
    print(f"Runtime ottimizzato:   {res_opt['runtime']:.4f} s")

    if res_base["objective"] == res_opt["objective"]:
        print("\nOK: i due modelli hanno lo stesso valore obiettivo.")
    else:
        print("\nATTENZIONE: i due modelli hanno objective diverso.")

    return res_base, res_opt


if __name__ == "__main__":
    compare_once(
        n=20,
        C=100,
        gamma=1.0,
        seed=42,
        s_factor=1.0,
        duration_type="short",
        size_type="low",
        time_limit=1800
    )