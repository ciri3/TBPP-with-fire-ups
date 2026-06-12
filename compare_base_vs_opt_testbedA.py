import os
import csv
import time
import itertools
import statistics
import matplotlib.pyplot as plt

from testbed_generators.testbed_a_generator import generate_jobs

# --- Import Modelli Base ---
from modello1.model1_tbpp_fu import solve_model1
from modello2.model2_tbpp_fu import solve_model2
from modello3.model3_tbpp_fu import solve_model3

# --- Import Modelli Ottimizzati ---
from modello1.model1_optimized_tbpp_fu import solve_model1_optimized
from modello2.model2_optimized_tbpp_fu import solve_model2_optimized
from modello3.model3_optimized_tbpp_fu import solve_model3_optimized


# Tutti gli output verranno creati dentro questa cartella:
BASE_OUTPUT_DIR = os.path.join("scalabilityTests", "TestaATesta_TestbedA")

# ==========================================================================
# importante: regolare i valori del test modificando le variabili in main()
# ==========================================================================



# =====================================================================
# FUNZIONI DI SUPPORTO
# =====================================================================

def safe_runtime(result, time_limit):
    """
    Runtime da usare nelle medie.

    Gurobi status:
    2 = OPTIMAL
    9 = TIME_LIMIT

    Se il modello va in time limit uso il runtime restituito dal modello,
    se manca uso direttamente time_limit.
    """
    status = result.get("status")
    if status in [2, 9]:
        return result.get("runtime", time_limit)
    return time_limit


def safe_stat(result, possible_keys):
    """
    Cerca una statistica nel dizionario risultato usando più possibili nomi.
    Serve perché i tuoi solve_model possono usare chiavi diverse.
    """
    for key in possible_keys:
        if key in result:
            return result.get(key)
    return None


def run_model(model_name, model_function, jobs, C, gamma, time_limit):
    """
    Esegue un singolo modello su una singola istanza.

    Oltre al tempo, legge anche:
    - num_vars: numero di variabili create nel modello Gurobi
    - num_constrs: numero di vincoli creati nel modello Gurobi

    Questi valori vengono salvati solo se le funzioni solve_modelX li restituiscono.
    """
    wall_start = time.time()
    result = model_function(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=time_limit,
        verbose=False,
    )
    wall_time = time.time() - wall_start
    gurobi_time = safe_runtime(result, time_limit)

    return {
        "model": model_name,
        "status": result.get("status"),
        "objective": result.get("objective"),
        "runtime": gurobi_time,
        "wall_time": wall_time,
        "servers": result.get("servers_used") or result.get("servers"),
        "fireups": result.get("fireups") or result.get("num_fireups"),
        "num_vars": safe_stat(result, ["num_vars", "NumVars", "nvars", "n_variables"]),
        "num_constrs": safe_stat(result, ["num_constrs", "NumConstrs", "nconstrs", "n_constraints"]),
    }


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_runtime_plot(n_values, n_rows, dict_models, graph_png, title):
    plt.figure(figsize=(9, 6))
    markers = ["o", "s", "^", "x", "D", "*"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]

    for idx, model_name in enumerate(dict_models):
        y = []
        for n in n_values:
            match = [r for r in n_rows if r["n"] == n and r["model"] == model_name]
            y.append(match[0]["mean_runtime"] if match else None)

        plt.plot(
            list(n_values),
            y,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2,
            label=model_name,
        )

    plt.title(title)
    plt.xlabel("Numero di job n")
    plt.ylabel("Tempo medio di soluzione [s]")
    plt.xticks(list(n_values))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_png, dpi=300)
    plt.close()


def make_model_size_plot(n_values, n_rows, dict_models, graph_png, metric, ylabel, title):
    """
    Crea un grafico per variabili oppure vincoli.

    Se un modello non ha valori disponibili, viene saltato.
    Se meno di due modelli hanno valori disponibili, il grafico non viene creato:
    così evitiamo grafici con una sola linea che possono trarre in inganno.
    """
    plt.figure(figsize=(9, 6))
    markers = ["o", "s", "^", "x", "D", "*"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]

    plotted_models = 0

    for idx, model_name in enumerate(dict_models):
        y = []
        has_at_least_one_value = False

        for n in n_values:
            match = [r for r in n_rows if r["n"] == n and r["model"] == model_name]
            value = match[0][metric] if match else None
            y.append(value)
            if value is not None:
                has_at_least_one_value = True

        if not has_at_least_one_value:
            print(f"ATTENZIONE: {metric} non disponibile per {model_name}. Linea saltata.")
            continue

        plotted_models += 1
        plt.plot(
            list(n_values),
            y,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2,
            label=model_name,
        )

    if plotted_models < 2:
        plt.close()
        print(
            f"ATTENZIONE: grafico {os.path.basename(graph_png)} non creato: "
            f"servono almeno 2 modelli con valori disponibili per {metric}."
        )
        return False

    plt.title(title)
    plt.xlabel("Numero di job n")
    plt.ylabel(ylabel)
    plt.xticks(list(n_values))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_png, dpi=300)
    plt.close()
    return True


# =====================================================================
# ESPERIMENTO HEAD-TO-HEAD
# =====================================================================

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

    Produce dentro output_dir:
    - detailed.csv: una riga per ogni istanza e modello
    - group_means.csv: media per gruppo Testbed A
    - n_means.csv: media aggregata per numero di job n
    - scalability_runtime.png: grafico runtime medio
    - scalability_num_vars.png: grafico numero medio variabili, se disponibile per entrambi i modelli
    - scalability_num_constrs.png: grafico numero medio vincoli, se disponibile per entrambi i modelli
    """
    os.makedirs(output_dir, exist_ok=True)

    detailed_csv = os.path.join(output_dir, "detailed.csv")
    group_csv = os.path.join(output_dir, "group_means.csv")
    n_csv = os.path.join(output_dir, "n_means.csv")

    runtime_png = os.path.join(output_dir, "scalability_runtime.png")
    vars_png = os.path.join(output_dir, "scalability_num_vars.png")
    constrs_png = os.path.join(output_dir, "scalability_num_constrs.png")

    groups = list(itertools.product(n_values, s_factors, durations, sizes))
    detailed_rows = []
    group_rows = []

    print(f"\n{'=' * 70}")
    print(f"AVVIO SFIDA: {comparison_name.upper()}")
    print(f"Output dir: {output_dir}")
    print(f"{'=' * 70}")

    global_start = time.time()

    for n, s_factor, duration_type, size_type in groups:
        print(f"--- Gruppo: n={n}, s={s_factor}, dur={duration_type}, size={size_type} ---")

        group_times = {m: [] for m in dict_models}
        group_vars = {m: [] for m in dict_models}
        group_constrs = {m: [] for m in dict_models}
        group_optimal_count = {m: 0 for m in dict_models}
        group_timelimit_count = {m: 0 for m in dict_models}
        group_other_status_count = {m: 0 for m in dict_models}

        for seed in range(first_seed, first_seed + num_instances):
            jobs = generate_jobs(
                n=n,
                C=C,
                s_factor=s_factor,
                duration_type=duration_type,
                size_type=size_type,
                seed=seed,
            )

            for model_name, model_function in dict_models.items():
                res = run_model(model_name, model_function, jobs, C, gamma, time_limit)

                group_times[model_name].append(res["runtime"])

                if res["num_vars"] is not None:
                    group_vars[model_name].append(res["num_vars"])
                if res["num_constrs"] is not None:
                    group_constrs[model_name].append(res["num_constrs"])

                if res["status"] == 2:
                    group_optimal_count[model_name] += 1
                elif res["status"] == 9:
                    group_timelimit_count[model_name] += 1
                else:
                    group_other_status_count[model_name] += 1

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
                    "num_vars": res["num_vars"],
                    "num_constrs": res["num_constrs"],
                })

        for model_name in dict_models:
            avg_time = statistics.mean(group_times[model_name])
            avg_vars = statistics.mean(group_vars[model_name]) if group_vars[model_name] else None
            avg_constrs = statistics.mean(group_constrs[model_name]) if group_constrs[model_name] else None

            group_rows.append({
                "n": n,
                "s_factor": s_factor,
                "duration": duration_type,
                "size": size_type,
                "model": model_name,
                "mean_runtime": avg_time,
                "mean_num_vars": avg_vars,
                "mean_num_constrs": avg_constrs,
                "optimal_count": group_optimal_count[model_name],
                "timelimit_count": group_timelimit_count[model_name],
                "other_status_count": group_other_status_count[model_name],
                "num_instances": num_instances,
            })

            print(
                f"{model_name:<15} | "
                f"mean runtime = {avg_time:8.3f} s | "
                f"vars = {avg_vars if avg_vars is not None else 'NA'} | "
                f"constrs = {avg_constrs if avg_constrs is not None else 'NA'} | "
                f"opt = {group_optimal_count[model_name]}/{num_instances} | "
                f"TL = {group_timelimit_count[model_name]}/{num_instances}"
            )
        print()

        # Salvataggio progressivo, utile se interrompi l'esecuzione
        save_csv(detailed_csv, detailed_rows)
        save_csv(group_csv, group_rows)

    # -----------------------------------------------------------------
    # Aggregazione per n: qui si vede la scalabilità rispetto ai job
    # -----------------------------------------------------------------
    n_rows = []
    for n in n_values:
        for model_name in dict_models:
            subset = [r for r in detailed_rows if r["n"] == n and r["model"] == model_name]
            runtimes = [r["runtime"] for r in subset]
            statuses = [r["status"] for r in subset]
            vars_values = [r["num_vars"] for r in subset if r["num_vars"] is not None]
            constrs_values = [r["num_constrs"] for r in subset if r["num_constrs"] is not None]

            if runtimes:
                n_rows.append({
                    "n": n,
                    "model": model_name,
                    "mean_runtime": statistics.mean(runtimes),
                    "median_runtime": statistics.median(runtimes),
                    "min_runtime": min(runtimes),
                    "max_runtime": max(runtimes),
                    "mean_num_vars": statistics.mean(vars_values) if vars_values else None,
                    "mean_num_constrs": statistics.mean(constrs_values) if constrs_values else None,
                    "optimal_count": sum(1 for s in statuses if s == 2),
                    "timelimit_count": sum(1 for s in statuses if s == 9),
                    "other_status_count": sum(1 for s in statuses if s not in [2, 9]),
                    "num_runs": len(runtimes),
                })

    save_csv(detailed_csv, detailed_rows)
    save_csv(group_csv, group_rows)
    save_csv(n_csv, n_rows)

    # -----------------------------------------------------------------
    # Grafici finali
    # -----------------------------------------------------------------
    make_runtime_plot(
        n_values=n_values,
        n_rows=n_rows,
        dict_models=dict_models,
        graph_png=runtime_png,
        title=f"Testbed A - {comparison_name} - runtime medio",
    )

    make_model_size_plot(
        n_values=n_values,
        n_rows=n_rows,
        dict_models=dict_models,
        graph_png=vars_png,
        metric="mean_num_vars",
        ylabel="Numero medio di variabili",
        title=f"Testbed A - {comparison_name} - variabili create",
    )

    make_model_size_plot(
        n_values=n_values,
        n_rows=n_rows,
        dict_models=dict_models,
        graph_png=constrs_png,
        metric="mean_num_constrs",
        ylabel="Numero medio di vincoli",
        title=f"Testbed A - {comparison_name} - vincoli creati",
    )

    print(f"Sfida completata in {(time.time() - global_start) / 60:.1f} min")
    print(f"Dati e grafici salvati in: {output_dir}")


# =====================================================================
# WRAPPERS PER UNIFICARE LE FIRME DELLE FUNZIONI
# =====================================================================

def wrap_m1_base(jobs, C, gamma, time_limit, verbose):
    # SALVAVITA: evita OOM su M1 base.
    # Con N_VALS=(3,5,7,10) non dovrebbe scattare.
    #if len(jobs) > 25:
    #    return {
    #        "status": 9,
    #        "runtime": time_limit,
    #        "objective": None,
    #        "servers_used": None,
    #        "fireups": None,
    #        "num_vars": None,
    #        "num_constrs": None,
    #    }
    return solve_model1(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=time_limit,
        verbose=verbose,
        binary_w=True,
    )


def wrap_m2_base(jobs, C, gamma, time_limit, verbose):
    return solve_model2(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=time_limit,
        verbose=verbose,
        binary_w=True,
    )


def wrap_m3_base(jobs, C, gamma, time_limit, verbose):
    return solve_model3(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=time_limit,
        verbose=verbose,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():
    # Parametri ridotti per confronto testa-a-testa.
    # Se vuoi replicare più fedelmente il paper, usa (50, 100, 150, 200),
    # ma con M1 base diventa facilmente pesante.
    N_VALS = ( 15, 30, 45, 60) #il modello 1 base dal 75 in su diventa eccessivamente pesante 
    NUM_INST = 5
    TIME_LIM = 900

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # --- SFIDA 1: M1 Base vs M1 Opt ---
    run_head_to_head_comparison(
        comparison_name="Modello 1 (Base vs Ottimizzato)",
        dict_models={
            "M1 Base": wrap_m1_base,
            "M1 Opt": solve_model1_optimized,
        },
        output_dir=os.path.join(BASE_OUTPUT_DIR, "results_testbedA_M1_vs_M1opt"),
        n_values=N_VALS,
        num_instances=NUM_INST,
        time_limit=TIME_LIM,
    )

    # --- SFIDA 2: M2 Base vs M2 Opt ---
    run_head_to_head_comparison(
        comparison_name="Modello 2 (Base vs Ottimizzato)",
        dict_models={
            "M2 Base": wrap_m2_base,
            "M2 Opt": solve_model2_optimized,
        },
        output_dir=os.path.join(BASE_OUTPUT_DIR, "results_testbedA_M2_vs_M2opt"),
        n_values=N_VALS,
        num_instances=NUM_INST,
        time_limit=TIME_LIM,
    )

    # --- SFIDA 3: M3 Base vs M3 Opt ---
    run_head_to_head_comparison(
        comparison_name="Modello 3 (Base vs Ottimizzato)",
        dict_models={
            "M3 Base": wrap_m3_base,
            "M3 Opt": solve_model3_optimized,
        },
        output_dir=os.path.join(BASE_OUTPUT_DIR, "results_testbedA_M3_vs_M3opt"),
        n_values=N_VALS,
        num_instances=NUM_INST,
        time_limit=TIME_LIM,
    )

    print("\nTUTTI I CONFRONTI TESTA A TESTA SONO STATI COMPLETATI!")
    print(f"Cartella principale risultati: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()


# =====================================================================
# NOTA IMPORTANTE SULLE STATISTICHE DEL MODELLO
# =====================================================================
# Per creare i grafici su variabili e vincoli, ogni solve_model deve
# restituire nel dizionario finale anche queste chiavi:
#
#     "num_vars": model.NumVars,
#     "num_constrs": model.NumConstrs,
#
# Queste righe vanno aggiunte dentro ogni file modello, dopo model.optimize()
# e prima del return finale.
#
# Se queste chiavi mancano in uno dei due modelli del confronto, il CSV viene
# comunque creato, ma il grafico variabili/vincoli viene creato solo se entrambi
# i modelli hanno dati disponibili.
# =====================================================================
