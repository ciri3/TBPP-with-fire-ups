import os
import csv
import time
import statistics
import matplotlib.pyplot as plt

from testbed_generators.testbed_b_generator import TestbedBGenerator
from modello1.model1_optimized_tbpp_fu import solve_model1_optimized
from modello2.model2_optimized_tbpp_fu import solve_model2_optimized
from modello3.model3_optimized_tbpp_fu import solve_model3_optimized


# ============================================================
# Scalability test - Testbed B / Category B del paper
# Confronto unico tra M1, M2, M3 ottimizzati
# Output:
#     scalabilityTests/confronto_modelli_opt_testbedB
# ============================================================


# ==========================================================================
# importante: regolare i valori del test modificando le variabili run_scalability_testbedB_optimized
# ==========================================================================


BASE_OUTPUT_DIR = os.path.join("scalabilityTests", "confronto_modelli_opt_testbedB")


# ============================================================
# Funzioni di supporto
# ============================================================

def safe_runtime(result, time_limit):
    """
    Runtime usato nelle medie.

    Gurobi status:
    2 = OPTIMAL
    9 = TIME_LIMIT

    Se il modello va in time limit, uso comunque il runtime restituito
    dal modello; se manca, uso direttamente time_limit.
    """
    status = result.get("status")
    if status in [2, 9]:
        return result.get("runtime", time_limit)
    return time_limit


def safe_stat(result, possible_keys):
    """
    Legge una statistica dal dizionario risultato usando nomi alternativi.
    Serve perché i diversi solve_model possono restituire chiavi con nomi diversi.
    """
    for key in possible_keys:
        if key in result:
            return result.get(key)
    return None


def run_model(model_name, model_function, jobs, C, gamma, time_limit):
    """
    Esegue un singolo modello su una singola istanza Testbed B.

    Oltre al runtime, prova a leggere:
    - num_vars: numero di variabili del modello Gurobi
    - num_constrs: numero di vincoli del modello Gurobi
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


def make_runtime_plot(T_values, T_rows, models, graph_png):
    plt.figure(figsize=(10, 6))

    markers = ["o", "s", "^"]
    linestyles = ["-", "--", "-."]

    for idx, model_name in enumerate(models):
        y = []
        for T_size in T_values:
            match = [r for r in T_rows if r["T_size"] == T_size and r["model"] == model_name]
            y.append(match[0]["mean_runtime"] if match else None)

        plt.plot(
            list(T_values),
            y,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2,
            label=model_name,
        )

    plt.title("Testbed B - confronto modelli ottimizzati - runtime medio")
    plt.xlabel("Numero di time step |T|")
    plt.ylabel("Tempo medio di soluzione [s]")
    plt.xticks(list(T_values))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_png, dpi=300)
    plt.close()


def make_model_size_plot(T_values, T_rows, models, graph_png, metric, ylabel, title):
    """
    Crea il grafico di variabili oppure vincoli.

    Se un modello non restituisce la statistica richiesta, la sua linea viene saltata.
    Se meno di due modelli hanno dati disponibili, il grafico non viene creato.
    """
    plt.figure(figsize=(10, 6))

    markers = ["o", "s", "^"]
    linestyles = ["-", "--", "-."]
    plotted_models = 0

    for idx, model_name in enumerate(models):
        y = []
        has_value = False

        for T_size in T_values:
            match = [r for r in T_rows if r["T_size"] == T_size and r["model"] == model_name]
            value = match[0][metric] if match else None
            y.append(value)
            if value is not None:
                has_value = True

        if not has_value:
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
    plt.xlabel("Numero di time step |T|")
    plt.ylabel(ylabel)
    plt.xticks(list(T_values))
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(graph_png, dpi=300)
    plt.close()
    return True


# ============================================================
# Esperimento principale
# ============================================================

def run_scalability_testbedB_optimized(
    output_dir=BASE_OUTPUT_DIR,
    C=100,
    gamma=1.0,
    T_values=(5, 10, 15, 20, 25, 30),
    classes=("I", "III", "V", "VI", "IX"),
    num_instances=5,
    first_seed=42,
    time_limit=900,
):
    """
    Confronta in un unico esperimento i tre modelli ottimizzati sul Testbed B.

    Produce dentro scalabilityTests/confronto_modelli_opt_testbedB:
    - detailed.csv
    - group_means.csv
    - T_means.csv
    - scalability_runtime.png
    - scalability_num_vars.png, se i modelli restituiscono num_vars
    - scalability_num_constrs.png, se i modelli restituiscono num_constrs

    Per avvicinarti al Testbed B del paper puoi usare, ad esempio:
        T_values=(10, 15, 20, 30, 40, 50, 60)
        classes=("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")
        num_instances=10
        time_limit=1800

    Nota pratica:
        Le classi VIII, IX, X e valori grandi di |T| possono diventare pesanti.
    """
    os.makedirs(output_dir, exist_ok=True)

    detailed_csv = os.path.join(output_dir, "detailed.csv")
    group_csv = os.path.join(output_dir, "group_means.csv")
    T_csv = os.path.join(output_dir, "T_means.csv")

    runtime_png = os.path.join(output_dir, "scalability_runtime.png")
    vars_png = os.path.join(output_dir, "scalability_num_vars.png")
    constrs_png = os.path.join(output_dir, "scalability_num_constrs.png")

    models = {
        "M1 Opt": solve_model1_optimized,
        "M2 Opt": solve_model2_optimized,
        "M3 Opt": solve_model3_optimized,
    }

    detailed_rows = []
    group_rows = []

    print("=" * 78)
    print("SCALABILITY TEST - TESTBED B - CONFRONTO MODELLI OTTIMIZZATI")
    print("=" * 78)
    print(f"Output dir: {output_dir}")
    print(f"T_values: {list(T_values)}")
    print(f"Classes: {list(classes)}")
    print(f"Istanze per classe: {num_instances}")
    print(f"Time limit per modello/istanza: {time_limit} s")
    print("=" * 78)

    global_start = time.time()

    for T_size in T_values:
        for class_name in classes:
            print(f"\n--- Gruppo: |T|={T_size}, class={class_name} ---")

            group_times = {m: [] for m in models}
            group_vars = {m: [] for m in models}
            group_constrs = {m: [] for m in models}
            group_n_jobs = []
            group_optimal_count = {m: 0 for m in models}
            group_timelimit_count = {m: 0 for m in models}
            group_other_status_count = {m: 0 for m in models}

            for instance_id in range(num_instances):
                seed = TestbedBGenerator.stable_seed(
                    base_seed=first_seed,
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

                for model_name, model_function in models.items():
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

            mean_n_jobs = statistics.mean(group_n_jobs) if group_n_jobs else None

            for model_name in models:
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
                    "num_instances": num_instances,
                })

                print(
                    f"{model_name:<8} | "
                    f"mean n_jobs = {mean_n_jobs:6.2f} | "
                    f"mean runtime = {avg_time:8.3f} s | "
                    f"vars = {avg_vars if avg_vars is not None else 'NA'} | "
                    f"constrs = {avg_constrs if avg_constrs is not None else 'NA'} | "
                    f"opt = {group_optimal_count[model_name]}/{num_instances} | "
                    f"TL = {group_timelimit_count[model_name]}/{num_instances}"
                )

            # Salvataggio progressivo: utile se interrompi il run.
            save_csv(detailed_csv, detailed_rows)
            save_csv(group_csv, group_rows)

    # ------------------------------------------------------------
    # Aggregazione per numero di time step |T|
    # ------------------------------------------------------------
    T_rows = []

    for T_size in T_values:
        for model_name in models:
            subset = [r for r in detailed_rows if r["T_size"] == T_size and r["model"] == model_name]
            runtimes = [r["runtime"] for r in subset]
            statuses = [r["status"] for r in subset]
            n_jobs_values = [r["n_jobs"] for r in subset]
            vars_values = [r["num_vars"] for r in subset if r["num_vars"] is not None]
            constrs_values = [r["num_constrs"] for r in subset if r["num_constrs"] is not None]

            if runtimes:
                T_rows.append({
                    "T_size": T_size,
                    "model": model_name,
                    "mean_n_jobs": statistics.mean(n_jobs_values) if n_jobs_values else None,
                    "median_n_jobs": statistics.median(n_jobs_values) if n_jobs_values else None,
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

    # ------------------------------------------------------------
    # Grafici finali
    # ------------------------------------------------------------
    make_runtime_plot(
        T_values=T_values,
        T_rows=T_rows,
        models=models,
        graph_png=runtime_png,
    )

    make_model_size_plot(
        T_values=T_values,
        T_rows=T_rows,
        models=models,
        graph_png=vars_png,
        metric="mean_num_vars",
        ylabel="Numero medio di variabili",
        title="Testbed B - confronto modelli ottimizzati - variabili create",
    )

    make_model_size_plot(
        T_values=T_values,
        T_rows=T_rows,
        models=models,
        graph_png=constrs_png,
        metric="mean_num_constrs",
        ylabel="Numero medio di vincoli",
        title="Testbed B - confronto modelli ottimizzati - vincoli creati",
    )

    total_minutes = (time.time() - global_start) / 60

    print("\n" + "=" * 78)
    print(f"TEST COMPLETATO IN {total_minutes:.1f} MINUTI")
    print(f"CSV dettagliato:       {detailed_csv}")
    print(f"CSV medie per gruppo:  {group_csv}")
    print(f"CSV medie per |T|:     {T_csv}")
    print(f"Grafico runtime:       {runtime_png}")
    print(f"Grafico variabili:     {vars_png}")
    print(f"Grafico vincoli:       {constrs_png}")
    print("=" * 78)


if __name__ == "__main__":
    run_scalability_testbedB_optimized()


