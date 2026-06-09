import os
import csv
import statistics
import matplotlib.pyplot as plt

from JobSetRandomGenerator import generate_jobs
from modello1.model1_tbpp_fu import solve_model1
from modello1.model1_optimized_tbpp_fu import solve_model1_optimized
from modello2.model2_tbpp_fu import solve_model2
from modello3.model3_tbpp_fu import solve_model3


OUTPUT_DIR = "night_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_value(res, key):
    return res.get(key, None)


def is_solution_available(res):
    return res.get("objective") is not None


def run_single_instance(n, seed, C=100, gamma=1.0, time_limit=1800):
    jobs = generate_jobs(
        n=n,
        C=C,
        s_factor=1.0,
        duration_type="short",
        size_type="low",
        seed=seed
    )

    print(f"\n--- n={n}, seed={seed}, jobs={len(jobs)} ---")

    res_m1_base = solve_model1(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=time_limit,
        verbose=False,
        binary_w=True
    )

    res_m1_opt = solve_model1_optimized(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=time_limit,
        verbose=False
    )

    res_m2 = solve_model2(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=time_limit,
        verbose=False,
        binary_w=True
    )

    res_m3 = solve_model3(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=time_limit,
        verbose=False
    )

    obj_base = safe_value(res_m1_base, "objective")
    obj_opt = safe_value(res_m1_opt, "objective")
    obj_m2 = safe_value(res_m2, "objective")
    obj_m3 = safe_value(res_m3, "objective")

    base_vs_opt_ok = (
        obj_base is not None
        and obj_opt is not None
        and abs(obj_base - obj_opt) < 1e-6
    )

    opt_vs_m2_ok = (
        obj_opt is not None
        and obj_m2 is not None
        and abs(obj_opt - obj_m2) < 1e-6
    )

    opt_vs_m3_ok = (
        obj_opt is not None
        and obj_m3 is not None
        and abs(obj_opt - obj_m3) < 1e-6
    )

    print(
        f"M1 base: {res_m1_base['runtime']:.4f}s obj={obj_base} | "
        f"M1 opt: {res_m1_opt['runtime']:.4f}s obj={obj_opt} | "
        f"M2: {res_m2['runtime']:.4f}s obj={obj_m2} | "
        f"M3: {res_m3['runtime']:.4f}s obj={obj_m3}"
    )

    return {
        "n": n,
        "seed": seed,

        "m1_base_status": res_m1_base["status"],
        "m1_opt_status": res_m1_opt["status"],
        "m2_status": res_m2["status"],
        "m3_status": res_m3["status"],

        "m1_base_objective": obj_base,
        "m1_opt_objective": obj_opt,
        "m2_objective": obj_m2,
        "m3_objective": obj_m3,

        "m1_base_runtime": res_m1_base["runtime"],
        "m1_opt_runtime": res_m1_opt["runtime"],
        "m2_runtime": res_m2["runtime"],
        "m3_runtime": res_m3["runtime"],

        "m1_base_servers": safe_value(res_m1_base, "servers_used"),
        "m1_opt_servers": safe_value(res_m1_opt, "servers_used"),
        "m2_servers": safe_value(res_m2, "servers_used"),
        "m3_servers": safe_value(res_m3, "servers_used"),

        "m1_base_fireups": safe_value(res_m1_base, "num_fireups"),
        "m1_opt_fireups": safe_value(res_m1_opt, "fireups"),
        "m2_fireups": safe_value(res_m2, "fireups"),
        "m3_fireups": safe_value(res_m3, "fireups"),

        "base_vs_opt_same_objective": base_vs_opt_ok,
        "opt_vs_m2_same_objective": opt_vs_m2_ok,
        "opt_vs_m3_same_objective": opt_vs_m3_ok,
    }


def save_csv(rows, filename):
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV salvato in: {path}")


def make_average_plot(rows):
    n_values = sorted(set(row["n"] for row in rows))

    avg_m1_base = []
    avg_m1_opt = []
    avg_m2 = []
    avg_m3 = []

    for n in n_values:
        subset = [row for row in rows if row["n"] == n]

        avg_m1_base.append(statistics.mean(row["m1_base_runtime"] for row in subset))
        avg_m1_opt.append(statistics.mean(row["m1_opt_runtime"] for row in subset))
        avg_m2.append(statistics.mean(row["m2_runtime"] for row in subset))
        avg_m3.append(statistics.mean(row["m3_runtime"] for row in subset))

    plt.figure(figsize=(12, 7))

    plt.plot(n_values, avg_m1_base, marker="o", label="Model 1 base")
    plt.plot(n_values, avg_m1_opt, marker="s", label="Model 1 ottimizzato")
    plt.plot(n_values, avg_m2, marker="^", label="Model 2")
    plt.plot(n_values, avg_m3, marker="x", label="Model 3")

    plt.title("Confronto runtime medio TBPP-FU")
    plt.xlabel("Numero di job n")
    plt.ylabel("Runtime medio [s]")
    plt.xticks(n_values)
    plt.grid(True, linestyle=":")
    plt.legend()

    path = os.path.join(OUTPUT_DIR, "runtime_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot salvato in: {path}")


def make_speedup_plot(rows):
    n_values = sorted(set(row["n"] for row in rows))

    speedups = []

    for n in n_values:
        subset = [row for row in rows if row["n"] == n]

        avg_base = statistics.mean(row["m1_base_runtime"] for row in subset)
        avg_opt = statistics.mean(row["m1_opt_runtime"] for row in subset)

        if avg_opt > 0:
            speedups.append(avg_base / avg_opt)
        else:
            speedups.append(None)

    plt.figure(figsize=(12, 7))
    plt.plot(n_values, speedups, marker="o", label="Speedup M1 base / M1 ottimizzato")

    plt.title("Speedup del Model 1 ottimizzato rispetto al Model 1 base")
    plt.xlabel("Numero di job n")
    plt.ylabel("Speedup")
    plt.xticks(n_values)
    plt.grid(True, linestyle=":")
    plt.legend()

    path = os.path.join(OUTPUT_DIR, "model1_speedup.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot speedup salvato in: {path}")


def print_correctness_summary(rows):
    total = len(rows)

    base_opt_ok = sum(row["base_vs_opt_same_objective"] for row in rows)
    opt_m2_ok = sum(row["opt_vs_m2_same_objective"] for row in rows)
    opt_m3_ok = sum(row["opt_vs_m3_same_objective"] for row in rows)

    print("\n========== CONTROLLO COERENZA ==========")
    print(f"M1 base vs M1 ottimizzato: {base_opt_ok}/{total} objective uguali")
    print(f"M1 ottimizzato vs M2:      {opt_m2_ok}/{total} objective uguali")
    print(f"M1 ottimizzato vs M3:      {opt_m3_ok}/{total} objective uguali")

    inconsistent = [
        row for row in rows
        if not row["base_vs_opt_same_objective"]
        or not row["opt_vs_m2_same_objective"]
        or not row["opt_vs_m3_same_objective"]
    ]

    if inconsistent:
        print("\nATTENZIONE: ci sono istanze con objective diversi.")
        print("Controlla il CSV, soprattutto status e gap.")
    else:
        print("\nOK: tutte le istanze hanno objective coerenti.")


def main():
    C = 100
    gamma = 1.0
    time_limit = 1800

    # Per la notte puoi aumentare.
    # Con licenza restricted, tieniti inizialmente prudente.
    n_values = [10, 20, 30, 40, 50, 60, 70, 80]
    seeds = range(42, 45)  # 3 istanze per ogni n, puoi aumentare se vuoi

    rows = []

    print("========== ESPERIMENTI NOTTURNI TBPP-FU ==========")
    print(f"n_values = {list(n_values)}")
    print(f"seeds = {list(seeds)}")
    print(f"time_limit = {time_limit} s per modello")
    print("==================================================")

    for n in n_values:
        for seed in seeds:
            row = run_single_instance(
                n=n,
                seed=seed,
                C=C,
                gamma=gamma,
                time_limit=time_limit
            )
            rows.append(row)

            # Salvataggio progressivo, così se interrompi non perdi tutto
            save_csv(rows, "partial_results.csv")

    save_csv(rows, "final_results.csv")
    make_average_plot(rows)
    make_speedup_plot(rows)
    print_correctness_summary(rows)

    print("\nEsperimenti completati.")


if __name__ == "__main__":
    main()