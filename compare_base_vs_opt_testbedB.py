import os
import csv
import time
import statistics
import matplotlib.pyplot as plt

from testbed_generators.testbed_b_generator import TestbedBGenerator

# --- Import Modelli Base ---
from modello1.model1_tbpp_fu import solve_model1
from modello2.model2_tbpp_fu import solve_model2
from modello3.model3_tbpp_fu import solve_model3

# --- Import Modelli Ottimizzati ---
from modello1.model1_optimized_tbpp_fu import solve_model1_optimized
from modello2.model2_optimized_tbpp_fu import solve_model2_optimized
from modello3.model3_optimized_tbpp_fu import solve_model3_optimized


# Tutti gli output verranno creati dentro questa cartella:
BASE_OUTPUT_DIR = os.path.join("scalabilityTests", "TestaATesta_TestbedB")

# ----------------------------------------------------------------------------
# importante: regolare i valori del test modificando le variabili in main()
# ----------------------------------------------------------------------------



# ---------------------------------------------------------------------
# FUNZIONI DI SUPPORTO
# ---------------------------------------------------------------------

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
    Serve perché i solve_model possono usare chiavi diverse.
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


def make_runtime_plot(T_values, T_rows, dict_models, graph_png, title):
    plt.figure(figsize=(9, 6))
    markers = ["o", "s", "^", "x", "D", "*"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]

    for idx, model_name in enumerate(dict_models):
        y = []
        for T_size in T_values:
            match = [
                r for r in T_rows
                if r["T_size"] == T_size and r["model"] == model_name
            ]
            y.append(match[0]["mean_runtime"] if match else None)

        plt.plot(
            list(T_values),
            y,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2,
            label=model_name,
        )

    plt.title(title)
    plt.xlabel("Numero di istanti temporali |T|")
    plt.ylabel("Tempo medio di soluzione [s]")
    plt.xticks(list(T_values))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_png, dpi=300)
    plt.close()


def make_model_size_plot(T_values, T_rows, dict_models, graph_png, metric, ylabel, title):
    """
    Crea un grafico per variabili oppure vincoli.

    Se un modello non ha valori disponibili, viene saltato.
    Se meno di due modelli hanno valori disponibili, il grafico non viene creato.
    """
    plt.figure(figsize=(9, 6))
    markers = ["o", "s", "^", "x", "D", "*"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]

    plotted_models = 0

    for idx, model_name in enumerate(dict_models):
        y = []
        has_at_least_one_value = False

        for T_size in T_values:
            match = [
                r for r in T_rows
                if r["T_size"] == T_size and r["model"] == model_name
            ]
            value = match[0][metric] if match else None
            y.append(value)
            if value is not None:
                has_at_least_one_value = True

        if not has_at_least_one_value:
            print(f"ATTENZIONE: {metric} non disponibile per {model_name}. Linea saltata.")
            continue

        plotted_models += 1
        plt.plot(
            list(T_values),
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
    plt.xlabel("Numero di istanti temporali |T|")
    plt.ylabel(ylabel)
    plt.xticks(list(T_values))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_png, dpi=300)
    plt.close()
    return True


def make_mean_jobs_plot(T_values, T_rows, graph_png, title):
    """
    Grafico del numero medio di job generati dal Testbed B.

    Nel Testbed B non scegli direttamente n.
    Scegli |T| e classe; il numero di job nasce dalla costruzione casuale.
    """
    plt.figure(figsize=(9, 6))

    y = []
    for T_size in T_values:
        match = [r for r in T_rows if r["T_size"] == T_size]
        if match:
            # T_rows contiene una riga per ogni modello; il numero medio di job
            # è uguale per i due modelli dello stesso confronto.
            y.append(match[0]["mean_n_jobs"])
        else:
            y.append(None)

    plt.plot(list(T_values), y, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Numero di istanti temporali |T|")
    plt.ylabel("Numero medio di job generati")
    plt.xticks(list(T_values))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(graph_png, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# ESPERIMENTO HEAD-TO-HEAD SU TESTBED B
# ---------------------------------------------------------------------

def run_head_to_head_comparison_testbedB(
    comparison_name,
    dict_models,
    output_dir,
    C=100,
    gamma=1.0,
    T_values=(5, 10, 15, 20),
    classes=("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"),
    instances_per_class=3,
    base_seed=42,
    time_limit=900,
):
    """
    Esegue il Testbed B confrontando esattamente i modelli passati in dict_models.

    Produce dentro output_dir:
    - detailed.csv: una riga per ogni istanza e modello
    - group_means.csv: media per gruppo (|T|, classe)
    - T_means.csv: media aggregata per |T|
    - scalability_runtime.png: grafico runtime medio rispetto a |T|
    - scalability_num_vars.png: grafico numero medio variabili, se disponibile
    - scalability_num_constrs.png: grafico numero medio vincoli, se disponibile
    - scalability_mean_jobs.png: grafico numero medio di job generati rispetto a |T|

    """
    os.makedirs(output_dir, exist_ok=True)

    detailed_csv = os.path.join(output_dir, "detailed.csv")
    group_csv = os.path.join(output_dir, "group_means.csv")
    T_csv = os.path.join(output_dir, "T_means.csv")

    runtime_png = os.path.join(output_dir, "scalability_runtime.png")
    vars_png = os.path.join(output_dir, "scalability_num_vars.png")
    constrs_png = os.path.join(output_dir, "scalability_num_constrs.png")
    mean_jobs_png = os.path.join(output_dir, "scalability_mean_jobs.png")

    detailed_rows = []
    group_rows = []

    print(f"\n{'=' * 70}")
    print(f"AVVIO SFIDA TESTBED B: {comparison_name.upper()}")
    print(f"T_values: {list(T_values)}")
    print(f"Classes: {list(classes)}")
    print(f"Instances per class: {instances_per_class}")
    print(f"Output dir: {output_dir}")
    print(f"{'=' * 70}")

    global_start = time.time()

    for T_size in T_values:
        if T_size < 2:
            raise ValueError("T_values deve contenere solo valori >= 2.")

        for class_name in classes:
            if class_name not in TestbedBGenerator.CLASS_PARAMS:
                raise ValueError(f"Classe non valida: {class_name}")

            print(f"--- Gruppo: |T|={T_size}, class={class_name} ---")

            group_times = {m: [] for m in dict_models}
            group_vars = {m: [] for m in dict_models}
            group_constrs = {m: [] for m in dict_models}
            group_optimal_count = {m: 0 for m in dict_models}
            group_timelimit_count = {m: 0 for m in dict_models}
            group_other_status_count = {m: 0 for m in dict_models}
            group_n_jobs = []

            for instance_id in range(instances_per_class):
                seed = TestbedBGenerator.stable_seed(
                    base_seed=base_seed,
                    T_size=T_size,
                    class_name=class_name,
                    instance_id=instance_id,
                )

                generator = TestbedBGenerator(seed=seed)
                jobs = generator.generate_instance(
                    T_size=T_size,
                    class_name=class_name,
                )
                n_jobs = len(jobs)
                group_n_jobs.append(n_jobs)

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
                        "T_size": T_size,
                        "class": class_name,
                        "instance_id": instance_id,
                        "seed": seed,
                        "n_jobs": n_jobs,
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

            mean_n_jobs = statistics.mean(group_n_jobs)

            for model_name in dict_models:
                avg_time = statistics.mean(group_times[model_name])
                avg_vars = statistics.mean(group_vars[model_name]) if group_vars[model_name] else None
                avg_constrs = statistics.mean(group_constrs[model_name]) if group_constrs[model_name] else None

                group_rows.append({
                    "T_size": T_size,
                    "class": class_name,
                    "model": model_name,
                    "mean_n_jobs": mean_n_jobs,
                    "mean_runtime": avg_time,
                    "mean_num_vars": avg_vars,
                    "mean_num_constrs": avg_constrs,
                    "optimal_count": group_optimal_count[model_name],
                    "timelimit_count": group_timelimit_count[model_name],
                    "other_status_count": group_other_status_count[model_name],
                    "num_instances": instances_per_class,
                })

                print(
                    f"{model_name:<15} | "
                    f"mean n_jobs = {mean_n_jobs:6.2f} | "
                    f"mean runtime = {avg_time:8.3f} s | "
                    f"vars = {avg_vars if avg_vars is not None else 'NA'} | "
                    f"constrs = {avg_constrs if avg_constrs is not None else 'NA'} | "
                    f"opt = {group_optimal_count[model_name]}/{instances_per_class} | "
                    f"TL = {group_timelimit_count[model_name]}/{instances_per_class}"
                )
            print()

            # Salvataggio progressivo, utile se interrompi l'esecuzione
            save_csv(detailed_csv, detailed_rows)
            save_csv(group_csv, group_rows)

    # -----------------------------------------------------------------
    # Aggregazione per |T|
    # -----------------------------------------------------------------
    T_rows = []
    for T_size in T_values:
        for model_name in dict_models:
            subset = [
                r for r in detailed_rows
                if r["T_size"] == T_size and r["model"] == model_name
            ]
            runtimes = [r["runtime"] for r in subset]
            statuses = [r["status"] for r in subset]
            n_jobs_values = [r["n_jobs"] for r in subset]
            vars_values = [r["num_vars"] for r in subset if r["num_vars"] is not None]
            constrs_values = [r["num_constrs"] for r in subset if r["num_constrs"] is not None]

            if runtimes:
                T_rows.append({
                    "T_size": T_size,
                    "model": model_name,
                    "mean_n_jobs": statistics.mean(n_jobs_values),
                    "median_n_jobs": statistics.median(n_jobs_values),
                    "min_n_jobs": min(n_jobs_values),
                    "max_n_jobs": max(n_jobs_values),
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
    save_csv(T_csv, T_rows)

    # -----------------------------------------------------------------
    # Grafici finali
    # -----------------------------------------------------------------
    make_runtime_plot(
        T_values=T_values,
        T_rows=T_rows,
        dict_models=dict_models,
        graph_png=runtime_png,
        title=f"Testbed B - {comparison_name} - runtime medio",
    )

    make_model_size_plot(
        T_values=T_values,
        T_rows=T_rows,
        dict_models=dict_models,
        graph_png=vars_png,
        metric="mean_num_vars",
        ylabel="Numero medio di variabili",
        title=f"Testbed B - {comparison_name} - variabili create",
    )

    make_model_size_plot(
        T_values=T_values,
        T_rows=T_rows,
        dict_models=dict_models,
        graph_png=constrs_png,
        metric="mean_num_constrs",
        ylabel="Numero medio di vincoli",
        title=f"Testbed B - {comparison_name} - vincoli creati",
    )

    make_mean_jobs_plot(
        T_values=T_values,
        T_rows=T_rows,
        graph_png=mean_jobs_png,
        title=f"Testbed B - {comparison_name} - numero medio di job generati",
    )

    print(f"Sfida completata in {(time.time() - global_start) / 60:.1f} min")
    print(f"Dati e grafici salvati in: {output_dir}")


# ---------------------------------------------------------------------
# WRAPPERS PER UNIFICARE LE FIRME DELLE FUNZIONI
# ---------------------------------------------------------------------

def wrap_m1_base(jobs, C, gamma, time_limit, verbose):
    
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


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    # Parametri prudenti.
    # Il Testbed B può generare più job di quanto ti aspetti, soprattutto
    # nelle classi VIII, IX, X oppure con |T| grande.
    T_VALS = (5,10,15,20)  
    CLASSES = ("I", "V", "VI", "IX")
    INSTANCES_PER_CLASS = 5
    TIME_LIM = 900

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # --- SFIDA 1: M1 Base vs M1 Opt ---
    run_head_to_head_comparison_testbedB(
        comparison_name="Modello 1 (Base vs Ottimizzato)",
        dict_models={
            "M1 Base": wrap_m1_base,
            "M1 Opt": solve_model1_optimized,
        },
        output_dir=os.path.join(BASE_OUTPUT_DIR, "results_testbedB_M1_vs_M1opt"),
        T_values=T_VALS,
        classes=CLASSES,
        instances_per_class=INSTANCES_PER_CLASS,
        time_limit=TIME_LIM,
    )

    # --- SFIDA 2: M2 Base vs M2 Opt ---
    run_head_to_head_comparison_testbedB(
        comparison_name="Modello 2 (Base vs Ottimizzato)",
        dict_models={
            "M2 Base": wrap_m2_base,
            "M2 Opt": solve_model2_optimized,
        },
        output_dir=os.path.join(BASE_OUTPUT_DIR, "results_testbedB_M2_vs_M2opt"),
        T_values=T_VALS,
        classes=CLASSES,
        instances_per_class=INSTANCES_PER_CLASS,
        time_limit=TIME_LIM,
    )

    # --- SFIDA 3: M3 Base vs M3 Opt ---
    run_head_to_head_comparison_testbedB(
        comparison_name="Modello 3 (Base vs Ottimizzato)",
        dict_models={
            "M3 Base": wrap_m3_base,
            "M3 Opt": solve_model3_optimized,
        },
        output_dir=os.path.join(BASE_OUTPUT_DIR, "results_testbedB_M3_vs_M3opt"),
        T_values=T_VALS,
        classes=CLASSES,
        instances_per_class=INSTANCES_PER_CLASS,
        time_limit=TIME_LIM,
    )

    print("\nTUTTI I CONFRONTI TESTA A TESTA SU TESTBED B SONO STATI COMPLETATI!")
    print(f"Cartella principale risultati: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()


