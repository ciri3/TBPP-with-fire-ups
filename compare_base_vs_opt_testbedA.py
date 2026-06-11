import os
import csv
import time
import itertools
import statistics
import matplotlib.pyplot as plt

from JobSetRandomGenerator import generate_jobs

# --- Import Modelli Base ---
from modello1.model1_tbpp_fu import solve_model1
from modello2.model2_tbpp_fu import solve_model2
from modello3.model3_tbpp_fu import solve_model3

# --- Import Modelli Ottimizzati ---
from modello1.model1_optimized_tbpp_fu import solve_model1_optimized
from modello2.model2_optimized_tbpp_fu import solve_model2_optimized
from modello3.model3_optimized_tbpp_fu import solve_model3_optimized


def safe_runtime(result, time_limit):
    status = result.get("status")
    if status in [2, 9]:  
        return result.get("runtime", time_limit)
    return time_limit


def run_model(model_name, model_function, jobs, C, gamma, time_limit):
    wall_start = time.time()
    result = model_function(
        jobs=jobs, C=C, gamma=gamma, time_limit=time_limit, verbose=False
    )
    wall_time = time.time() - wall_start
    gurobi_time = safe_runtime(result, time_limit)

    return {
        "model": model_name,
        "status": result.get("status"),
        "objective": result.get("objective"),
        "runtime": gurobi_time,
        "wall_time": wall_time,
        "servers": result.get("servers_used") or result.get("servers"), # Compatibilità chiavi
        "fireups": result.get("fireups"),
    }


def run_head_to_head_comparison(
    comparison_name,
    dict_models,
    output_dir,
    C=100,
    gamma=1.0,
    n_values=(50, 100, 150, 200),
    s_factors=(1.0, 1.2),
    durations=("short", "long"),
    sizes=("low", "high"),
    num_instances=5,
    first_seed=42,
    time_limit=1800,
):
    """
    Esegue il Testbed A confrontando esattamente i modelli passati in dict_models.
    """
    os.makedirs(output_dir, exist_ok=True)

    detailed_csv = os.path.join(output_dir, f"detailed.csv")
    group_csv = os.path.join(output_dir, f"group_means.csv")
    n_csv = os.path.join(output_dir, f"n_means.csv")
    graph_png = os.path.join(output_dir, f"scalability_plot.png")

    groups = list(itertools.product(n_values, s_factors, durations, sizes))
    detailed_rows = []
    group_rows = []

    print(f"\n{'='*60}")
    print(f" AVVIO SFIDA: {comparison_name.upper()}")
    print(f"{'='*60}")

    global_start = time.time()

    for n, s_factor, duration_type, size_type in groups:
        print(f"--- Gruppo: n={n}, s={s_factor}, dur={duration_type}, size={size_type} ---")

        group_times = {m: [] for m in dict_models}
        group_optimal_count = {m: 0 for m in dict_models}
        group_timelimit_count = {m: 0 for m in dict_models}

        for seed in range(first_seed, first_seed + num_instances):
            jobs = generate_jobs(n=n, C=C, s_factor=s_factor, duration_type=duration_type, size_type=size_type, seed=seed)

            for model_name, model_function in dict_models.items():
                res = run_model(model_name, model_function, jobs, C, gamma, time_limit)

                group_times[model_name].append(res["runtime"])

                if res["status"] == 2:
                    group_optimal_count[model_name] += 1
                elif res["status"] == 9:
                    group_timelimit_count[model_name] += 1

                detailed_rows.append({
                    "n": n, "s_factor": s_factor, "duration": duration_type, "size": size_type, "seed": seed,
                    "model": model_name, "status": res["status"], "objective": res["objective"],
                    "runtime": res["runtime"], "wall_time": res["wall_time"],
                    "servers": res["servers"], "fireups": res["fireups"],
                })

        for model_name in dict_models:
            avg_time = statistics.mean(group_times[model_name])
            group_rows.append({
                "n": n, "s_factor": s_factor, "duration": duration_type, "size": size_type,
                "model": model_name, "mean_runtime": avg_time,
                "optimal_count": group_optimal_count[model_name],
                "timelimit_count": group_timelimit_count[model_name], "num_instances": num_instances,
            })
            print(f"{model_name:<15} | mean runtime = {avg_time:8.3f} s | opt = {group_optimal_count[model_name]}/{num_instances} | TL = {group_timelimit_count[model_name]}/{num_instances}")
        print()

    # --- Salvataggi CSV ---
    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)

    with open(group_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    n_rows = []
    for n in n_values:
        for model_name in dict_models:
            runtimes = [r["runtime"] for r in detailed_rows if r["n"] == n and r["model"] == model_name]
            statuses = [r["status"] for r in detailed_rows if r["n"] == n and r["model"] == model_name]
            if runtimes:
                n_rows.append({
                    "n": n, "model": model_name, "mean_runtime": statistics.mean(runtimes),
                    "median_runtime": statistics.median(runtimes), "min_runtime": min(runtimes), "max_runtime": max(runtimes),
                    "optimal_count": sum(1 for s in statuses if s == 2), "timelimit_count": sum(1 for s in statuses if s == 9),
                    "num_runs": len(runtimes),
                })

    with open(n_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(n_rows[0].keys()))
        writer.writeheader()
        writer.writerows(n_rows)

    # --- Grafico ---
    plt.figure(figsize=(9, 6))
    markers = ["o", "s"]
    linestyles = ["-", "--"]
    colors = ["red", "green"]

    for idx, model_name in enumerate(dict_models):
        y = [r["mean_runtime"] for r in n_rows if r["model"] == model_name]
        plt.plot(list(n_values), y, marker=markers[idx], linestyle=linestyles[idx], color=colors[idx], linewidth=2, label=model_name)

    plt.title(f"Testbed A - {comparison_name}")
    plt.xlabel("Numero di job n")
    plt.ylabel("Tempo medio di soluzione (secondi)")
    plt.xticks(list(n_values))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_png, dpi=300)
    plt.close()

    print(f"Sfida completata in {(time.time() - global_start) / 60:.1f} min. Dati in: {output_dir}")


# =====================================================================
# WRAPPERS PER UNIFICARE LE FIRME DELLE FUNZIONI
# =====================================================================

def wrap_m1_base(jobs, C, gamma, time_limit, verbose):
    if len(jobs) > 25: # SALVAVITA: Evita OOM per M1 Base
        return {"status": 9, "runtime": time_limit, "objective": None, "servers_used": None, "fireups": None}
    return solve_model1(jobs=jobs, C=C, gamma=gamma, time_limit=time_limit, verbose=verbose, binary_w=True)

def wrap_m2_base(jobs, C, gamma, time_limit, verbose):
    return solve_model2(jobs=jobs, C=C, gamma=gamma, time_limit=time_limit, verbose=verbose, binary_w=True)

def wrap_m3_base(jobs, C, gamma, time_limit, verbose):
    return solve_model3(jobs=jobs, C=C, gamma=gamma, time_limit=time_limit, verbose=verbose)


def main():
    # NOSTRI PARAMETRI
    N_VALS = (5, 10, 15, 20)
    NUM_INST = 5
    TIME_LIM = 900

    # --- SFIDA 1: M1 Base vs M1 Opt ---
    run_head_to_head_comparison(
        comparison_name="Modello 1 (Base vs Ottimizzato)",
        dict_models={
            "M1 Base": wrap_m1_base,
            "M1 Opt": solve_model1_optimized
        },
        output_dir="results_testbedA_M1_vs_M1opt",
        n_values=N_VALS, num_instances=NUM_INST, time_limit=TIME_LIM
    )

    # --- SFIDA 2: M2 Base vs M2 Opt ---
    run_head_to_head_comparison(
        comparison_name="Modello 2 (Base vs Ottimizzato)",
        dict_models={
            "M2 Base": wrap_m2_base,
            "M2 Opt": solve_model2_optimized
        },
        output_dir="results_testbedA_M2_vs_M2opt",
        n_values=N_VALS, num_instances=NUM_INST, time_limit=TIME_LIM
    )

    # --- SFIDA 3: M3 Base vs M3 Opt ---
    run_head_to_head_comparison(
        comparison_name="Modello 3 (Base vs Ottimizzato)",
        dict_models={
            "M3 Base": wrap_m3_base,
            "M3 Opt": solve_model3_optimized
        },
        output_dir="results_testbedA_M3_vs_M3opt",
        n_values=N_VALS, num_instances=NUM_INST, time_limit=TIME_LIM
    )

    print("\nTUTTI I CONFRONTI TESTA A TESTA SONO STATI COMPLETATI!")

if __name__ == "__main__":
    main()