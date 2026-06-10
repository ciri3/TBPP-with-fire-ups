import os
import csv
import time
import itertools
import statistics
import matplotlib.pyplot as plt

from JobSetRandomGenerator import generate_jobs
from modello1.model1_optimized_tbpp_fu import solve_model1_optimized
from modello2.model2_optimized_tbpp_fu import solve_model2_optimized
from modello3.model3_optimized_tbpp_fu import solve_model3_optimized


# ============================================================
# Scalability test - Testbed A del paper
# Solo modelli ottimizzati M1, M2, M3
# ============================================================


def safe_runtime(result, time_limit):
    """
    Restituisce il tempo da usare nelle medie.

    Se Gurobi trova l'ottimo (status 2) oppure arriva al time limit
    con una soluzione/parziale informazione (status 9), usa il runtime
    restituito dal modello. Negli altri casi usa cautelativamente il
    time limit, così l'istanza problematica penalizza la media.
    """
    status = result.get("status")
    if status in [2, 9]:  # 2 = OPTIMAL, 9 = TIME_LIMIT in Gurobi
        return result.get("runtime", time_limit)
    return time_limit


def run_model(model_name, model_function, jobs, C, gamma, time_limit):
    """
    Esegue un modello ottimizzato e misura anche un tempo esterno Python.

    Il tempo usato per il grafico è il runtime restituito da Gurobi dentro
    al dizionario del modello, non il wall-clock Python esterno.
    wall_time è salvato nel CSV solo come controllo diagnostico.
    """
    wall_start = time.time()

    result = model_function(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=time_limit,
        verbose=False
    )

    wall_time = time.time() - wall_start
    gurobi_time = safe_runtime(result, time_limit)

    return {
        "model": model_name,
        "status": result.get("status"),
        "objective": result.get("objective"),
        "runtime": gurobi_time,
        "wall_time": wall_time,
        "servers": result.get("servers"),
        "fireups": result.get("fireups"),
    }


def run_scalability_testbedA_optimized(
    output_dir="results_testbedA_optimized",
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
    Esegue il Testbed A come nel paper:
    - C = 100
    - n in {50, 100, 150, 200}
    - s in {n, 1.2n}, implementato tramite s_factor in {1.0, 1.2}
    - durate short: [10, 30], long: [20, 60]
    - size low: [25, 50], high: [25, 75]
    - 5 istanze per ogni gruppo
    - time limit 1800 secondi

    Output:
    - CSV dettagliato: una riga per modello e istanza
    - CSV aggregato per gruppo: media sui 5 seed dello stesso gruppo
    - CSV aggregato per n: media su tutti i gruppi con lo stesso n
    - PNG finale: tempo medio in funzione di n
    """

    os.makedirs(output_dir, exist_ok=True)

    detailed_csv = os.path.join(output_dir, "testbedA_optimized_detailed.csv")
    group_csv = os.path.join(output_dir, "testbedA_optimized_group_means.csv")
    n_csv = os.path.join(output_dir, "testbedA_optimized_n_means.csv")
    graph_png = os.path.join(output_dir, "testbedA_optimized_scalability.png")

    models = {
        "M1 optimized": solve_model1_optimized,
        "M2 optimized": solve_model2_optimized,
        "M3 optimized": solve_model3_optimized,
    }

    groups = list(itertools.product(n_values, s_factors, durations, sizes))

    detailed_rows = []
    group_rows = []

    print("============================================================")
    print("SCALABILITY TEST - TESTBED A - SOLO MODELLI OTTIMIZZATI")
    print("============================================================")
    print(f"Gruppi: {len(groups)}")
    print(f"Istanze per gruppo: {num_instances}")
    print(f"Time limit per modello/istanza: {time_limit} s")
    print(f"Output dir: {output_dir}")
    print("============================================================\n")

    global_start = time.time()

    for n, s_factor, duration_type, size_type in groups:
        print(f"--- Gruppo: n={n}, s_factor={s_factor}, duration={duration_type}, size={size_type} ---")

        group_times = {model_name: [] for model_name in models}
        group_optimal_count = {model_name: 0 for model_name in models}
        group_timelimit_count = {model_name: 0 for model_name in models}

        for seed in range(first_seed, first_seed + num_instances):
            jobs = generate_jobs(
                n=n,
                C=C,
                s_factor=s_factor,
                duration_type=duration_type,
                size_type=size_type,
                seed=seed,
            )

            for model_name, model_function in models.items():
                res = run_model(
                    model_name=model_name,
                    model_function=model_function,
                    jobs=jobs,
                    C=C,
                    gamma=gamma,
                    time_limit=time_limit,
                )

                group_times[model_name].append(res["runtime"])

                if res["status"] == 2:
                    group_optimal_count[model_name] += 1
                elif res["status"] == 9:
                    group_timelimit_count[model_name] += 1

                detailed_rows.append({
                    "n": n,
                    "s_factor": s_factor,
                    "duration": duration_type,
                    "size": size_type,
                    "seed": seed,
                    "model": model_name,
                    "status": res["status"],
                    "objective": res["objective"],
                    "runtime": res["runtime"],
                    "wall_time": res["wall_time"],
                    "servers": res["servers"],
                    "fireups": res["fireups"],
                })

        for model_name in models:
            avg_time = statistics.mean(group_times[model_name])
            group_rows.append({
                "n": n,
                "s_factor": s_factor,
                "duration": duration_type,
                "size": size_type,
                "model": model_name,
                "mean_runtime": avg_time,
                "optimal_count": group_optimal_count[model_name],
                "timelimit_count": group_timelimit_count[model_name],
                "num_instances": num_instances,
            })
            print(
                f"{model_name:<13} | mean runtime = {avg_time:8.3f} s | "
                f"optimal = {group_optimal_count[model_name]}/{num_instances} | "
                f"time limit = {group_timelimit_count[model_name]}/{num_instances}"
            )

        print()

    # ------------------------------------------------------------
    # Salvataggio CSV dettagliato
    # ------------------------------------------------------------
    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_rows)

    # ------------------------------------------------------------
    # Salvataggio CSV aggregato per gruppo
    # ------------------------------------------------------------
    with open(group_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    # ------------------------------------------------------------
    # Aggregazione finale per n
    # Per ogni modello e per ogni n, media su TUTTE le istanze con quel n:
    # 2 s_factors * 2 durations * 2 sizes * 5 seed = 40 tempi per modello.
    # ------------------------------------------------------------
    n_rows = []
    for n in n_values:
        for model_name in models:
            runtimes = [
                row["runtime"]
                for row in detailed_rows
                if row["n"] == n and row["model"] == model_name
            ]
            statuses = [
                row["status"]
                for row in detailed_rows
                if row["n"] == n and row["model"] == model_name
            ]

            n_rows.append({
                "n": n,
                "model": model_name,
                "mean_runtime": statistics.mean(runtimes),
                "median_runtime": statistics.median(runtimes),
                "min_runtime": min(runtimes),
                "max_runtime": max(runtimes),
                "optimal_count": sum(1 for s in statuses if s == 2),
                "timelimit_count": sum(1 for s in statuses if s == 9),
                "num_runs": len(runtimes),
            })

    with open(n_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(n_rows[0].keys()))
        writer.writeheader()
        writer.writerows(n_rows)

    # ------------------------------------------------------------
    # Grafico finale: tempo medio vs numero di job
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    markers = {
        "M1 optimized": "o",
        "M2 optimized": "s",
        "M3 optimized": "^",
    }

    linestyles = {
        "M1 optimized": "-",
        "M2 optimized": "--",
        "M3 optimized": "-.",
    }

    for model_name in models:
        y = [
            row["mean_runtime"]
            for row in n_rows
            if row["model"] == model_name
        ]
        plt.plot(
            list(n_values),
            y,
            marker=markers[model_name],
            linestyle=linestyles[model_name],
            linewidth=2,
            label=model_name,
        )

    plt.title("Testbed A - Scalability test dei modelli ottimizzati")
    plt.xlabel("Numero di job n")
    plt.ylabel("Tempo medio di soluzione (secondi)")
    plt.xticks(list(n_values))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_png, dpi=300)
    plt.close()

    total_minutes = (time.time() - global_start) / 60

    print("============================================================")
    print(f"TEST COMPLETATO IN {total_minutes:.1f} MINUTI")
    print(f"CSV dettagliato:       {detailed_csv}")
    print(f"CSV medie per gruppo:  {group_csv}")
    print(f"CSV medie per n:       {n_csv}")
    print(f"Grafico finale:        {graph_png}")
    print("============================================================")


if __name__ == "__main__":
    run_scalability_testbedA_optimized()
