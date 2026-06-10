# modello3/model3_optimized_tbpp_fu.py

import math
import gurobipy as gp
from gurobipy import GRB


def intervals_overlap(job_a, job_b):
    """
    Controlla se due job si sovrappongono temporalmente.

    Intervalli considerati: [s, e)
    Due job si sovrappongono se:
        s_a < e_b and s_b < e_a
    """
    s_a, e_a, _ = job_a
    s_b, e_b, _ = job_b
    return s_a < e_b and s_b < e_a


def compute_material_lower_bound(sorted_jobs, C):
    """
    R0 / Lit-A adattato al Modello 3.

    Calcola h0 = ceil( max_t carico_totale_attivo(t) / C ).

    Basta controllare gli start time, perché il carico totale può aumentare
    solo quando inizia almeno un job; agli end time può solo diminuire.
    """
    start_times = sorted({job[0] for job in sorted_jobs})

    max_load = 0
    for t in start_times:
        load_t = sum(
            c_i
            for s_i, e_i, c_i in sorted_jobs
            if s_i <= t < e_i
        )
        max_load = max(max_load, load_t)

    return math.ceil(max_load / C)


def lift_item_sizes(sorted_jobs, C):
    """
    R3: lifting delle dimensioni dei job.

    Per ogni job i:
        A(i) = job che si sovrappongono temporalmente con i.

    Calcoliamo epsilon(i), cioè il massimo carico aggiuntivo che può essere
    messo insieme a i senza superare C.

    Poi aumentiamo c_i fino a:
        c_i_lifted = C - epsilon(i)

    Nota:
    - Questa modifica non cambia le soluzioni intere ammissibili.
    - Serve a rafforzare il rilassamento lineare.
    - Per istanze grandi può essere costosa, perché risolve un piccolo
      knapsack 0/1 per ogni job. Qui uso una DP semplice perché C è
      tipicamente 100 nel paper.
    """
    lifted_jobs = list(sorted_jobs)

    for i, job_i in enumerate(sorted_jobs):
        s_i, e_i, c_i = job_i

        # Job che si sovrappongono temporalmente con i
        overlapping_sizes = []
        for p, job_p in enumerate(sorted_jobs):
            if p == i:
                continue
            if intervals_overlap(job_i, job_p):
                overlapping_sizes.append(job_p[2])

        residual_capacity = C - c_i

        # Se non posso aggiungere nulla, il lifting non cambia nulla.
        if residual_capacity <= 0:
            continue

        # Knapsack 0/1: massima somma <= residual_capacity
        possible = [False] * (residual_capacity + 1)
        possible[0] = True

        for size in overlapping_sizes:
            if size > residual_capacity:
                continue
            for q in range(residual_capacity, size - 1, -1):
                if possible[q - size]:
                    possible[q] = True

        epsilon_i = max(q for q, ok in enumerate(possible) if ok)

        if epsilon_i < residual_capacity:
            c_i_lifted = C - epsilon_i
            lifted_jobs[i] = (s_i, e_i, c_i_lifted)

    return lifted_jobs


def solve_model3_optimized(
    jobs,
    C,
    gamma=1.0,
    time_limit=None,
    verbose=True,
    use_lifting=True,
    use_material_lb=True
):
    """
    Modello 3 ottimizzato per il Temporal Bin Packing Problem with Fire-Ups.

    Include:
    - R0: lower bound h0 sul numero di server inizializzatori
    - R1*: Delta_red, cioè eliminazione delle coppie (i,k) impossibili
    - R2*: disuguaglianza valida x[k,k] <= w[k]
    - R3: lifting delle capacità c_i
    - R5*: Delta_red_nd, cioè eliminazione dei vincoli di capacità dominati

    R4 NON viene applicata nel Modello 3:
    il paper dice esplicitamente che escludere i job con stesso start time
    da delta_plus_i porterebbe a contare fire-up doppi in alcuni casi.
    """

    # ------------------------------------------------------------
    # 1. Ordinamento dei job
    # ------------------------------------------------------------
    indexed_jobs = list(enumerate(jobs))
    indexed_jobs.sort(key=lambda item: (item[1][0], item[1][1], item[0]))

    original_ids = [idx for idx, job in indexed_jobs]
    sorted_jobs_original_c = [job for idx, job in indexed_jobs]

    n = len(sorted_jobs_original_c)
    I = range(n)

    # Controllo fattibilità elementare
    for i, (_, _, c_i) in enumerate(sorted_jobs_original_c):
        if c_i > C:
            raise ValueError(
                f"Job {i} ha richiesta c_i={c_i} maggiore della capacità C={C}."
            )

    # ------------------------------------------------------------
    # 2. R3: lifting delle dimensioni
    # ------------------------------------------------------------
    if use_lifting:
        sorted_jobs = lift_item_sizes(sorted_jobs_original_c, C)
    else:
        sorted_jobs = list(sorted_jobs_original_c)

    # ------------------------------------------------------------
    # 3. R0: lower bound h0 sul numero minimo di server
    # ------------------------------------------------------------
    if use_material_lb:
        h0 = compute_material_lower_bound(sorted_jobs_original_c, C)
    else:
        h0 = 0

    # ------------------------------------------------------------
    # 4. Costruzione di delta_i e delta_plus_i
    # ------------------------------------------------------------
    # delta usa gli start/end time, ma i coefficienti di capacità nei vincoli
    # useranno le capacità eventualmente liftate.
    delta = {}
    delta_plus = {}

    for i in I:
        s_i, _, _ = sorted_jobs_original_c[i]

        delta[i] = []
        delta_plus[i] = []

        for j in range(i):
            _, e_j, _ = sorted_jobs_original_c[j]

            if s_i < e_j:
                delta[i].append(j)

            # Nel Modello 3 NON applichiamo R4:
            # quindi NON escludiamo i job con stesso start time.
            if s_i <= e_j:
                delta_plus[i].append(j)

    # ------------------------------------------------------------
    # 5. R1*: costruzione di Delta_red
    # ------------------------------------------------------------
    # Delta base sarebbe: (i,k) con k <= i.
    # Delta_red elimina le coppie in cui i e k si sovrappongono e
    # c_i + c_k > C, perché non potrebbero mai stare sullo stesso server.
    Delta_red = set()

    for i in I:
        for k in range(i + 1):
            if i == k:
                Delta_red.add((i, k))
                continue

            job_i = sorted_jobs_original_c[i]
            job_k = sorted_jobs_original_c[k]

            c_i = sorted_jobs[i][2]      # capacità eventualmente liftata
            c_k = sorted_jobs[k][2]      # capacità eventualmente liftata

            if intervals_overlap(job_i, job_k) and c_i + c_k > C:
                continue

            Delta_red.add((i, k))

    # ------------------------------------------------------------
    # 6. R5*: costruzione di Delta_red_nd
    # ------------------------------------------------------------
    # Se i e j hanno stesso start time, k < i < j,
    # (i,k), (j,k) in Delta_red, allora (j,k) domina (i,k).
    #
    # Questa riduzione vale SOLO per i vincoli di capacità.
    Delta_red_nd = set()

    for (i, k) in Delta_red:
        # Le coppie diagonali non generano vincoli di capacità utili.
        # Le teniamo comunque nell'insieme per completezza/debug.
        if i == k:
            Delta_red_nd.add((i, k))
            continue

        s_i = sorted_jobs_original_c[i][0]
        dominated = False

        for j in range(i + 1, n):
            s_j = sorted_jobs_original_c[j][0]

            if s_j > s_i:
                break

            if s_j == s_i and (j, k) in Delta_red:
                dominated = True
                break

        if not dominated:
            Delta_red_nd.add((i, k))

    # ------------------------------------------------------------
    # 7. Creazione modello Gurobi
    # ------------------------------------------------------------
    model = gp.Model("TBPP_FU_Model3_Optimized")

    if not verbose:
        model.setParam("OutputFlag", 0)

    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)

    # ------------------------------------------------------------
    # 8. Variabili
    # ------------------------------------------------------------
    # x[i,k] = 1 se job i è eseguito sul server inizializzato da job k
    # w[i]   = 1 se job i causa un fire-up
    x = model.addVars(
        sorted(Delta_red),
        vtype=GRB.BINARY,
        name="x"
    )

    w = model.addVars(
        list(I),
        vtype=GRB.BINARY,
        name="w"
    )

    # ------------------------------------------------------------
    # 9. Funzione obiettivo
    # ------------------------------------------------------------
    model.setObjective(
        gamma * gp.quicksum(w[i] for i in I)
        + gp.quicksum(x[k, k] for k in I),
        GRB.MINIMIZE
    )

    # ------------------------------------------------------------
    # 10. R0: lower bound sui server inizializzatori
    # ------------------------------------------------------------
    # Nel Modello 3 non abbiamo z[k].
    # Il server usato/iniziato da k è rappresentato da x[k,k].
    #
    # h0 <= sum_k x[k,k]
    if h0 > 0:
        model.addConstr(
            gp.quicksum(x[k, k] for k in I) >= h0,
            name="material_lower_bound"
        )

    # ------------------------------------------------------------
    # 11. Vincoli di assegnamento (33)
    # ------------------------------------------------------------
    for i in I:
        feasible_initializers = [
            k for k in range(i + 1)
            if (i, k) in Delta_red
        ]

        model.addConstr(
            gp.quicksum(x[i, k] for k in feasible_initializers) == 1,
            name=f"assign_{i}"
        )

    # ------------------------------------------------------------
    # 12. Vincoli di capacità ridotti (32)
    # ------------------------------------------------------------
    # Creati solo per (i,k) in Delta_red_nd e i != k.
    for (i, k) in sorted(Delta_red_nd):
        if i == k:
            continue

        c_i = sorted_jobs[i][2]

        model.addConstr(
            gp.quicksum(
                sorted_jobs[j][2] * x[j, k]
                for j in delta[i]
                if (j, k) in Delta_red
            )
            + c_i * x[i, k]
            <= C * x[k, k],
            name=f"capacity_{i}_{k}"
        )

    # ------------------------------------------------------------
    # 13. Vincoli di collegamento (34)
    # ------------------------------------------------------------
    for (i, k) in sorted(Delta_red):
        if i != k:
            model.addConstr(
                x[i, k] <= x[k, k],
                name=f"link_{i}_{k}"
            )

    # ------------------------------------------------------------
    # 14. R2*: disuguaglianze valide x[k,k] <= w[k]
    # ------------------------------------------------------------
    # Se k inizializza un server, allora k causa sicuramente un fire-up.
    for k in I:
        model.addConstr(
            x[k, k] <= w[k],
            name=f"valid_initializer_fireup_{k}"
        )

    # ------------------------------------------------------------
    # 15. Vincoli sui fire-up (35)
    # ------------------------------------------------------------
    # Nota: delta_plus NON è ridotto come in R4 del Modello 2.
    for (i, k) in sorted(Delta_red):
        model.addConstr(
            gp.quicksum(
                x[j, k]
                for j in delta_plus[i]
                if (j, k) in Delta_red
            )
            - x[i, k]
            + w[i]
            >= 0,
            name=f"fireup_{i}_{k}"
        )

    # ------------------------------------------------------------
    # 16. Risoluzione
    # ------------------------------------------------------------
    model.optimize()

    # ------------------------------------------------------------
    # 17. Estrazione soluzione
    # ------------------------------------------------------------
    result = {
        "status": model.Status,
        "objective": None,
        "fireups": None,
        "servers_used": None,
        "assignment": None,
        "sorted_jobs": sorted_jobs_original_c,
        "lifted_jobs": sorted_jobs,
        "original_ids": original_ids,
        "runtime": model.Runtime,
        "h0": h0,
        "n_vars": model.NumVars,
        "n_constrs": model.NumConstrs,
        "n_nonzeros": model.NumNZs,
        "Delta_red_size": len(Delta_red),
        "Delta_red_nd_size": len(Delta_red_nd),
        "model": model,
    }

    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        if model.SolCount > 0:
            result["objective"] = model.ObjVal

            fireup_jobs = [
                i for i in I
                if w[i].X > 0.5
            ]

            initializer_jobs = [
                k for k in I
                if (k, k) in Delta_red and x[k, k].X > 0.5
            ]

            result["fireups"] = len(fireup_jobs)
            result["servers_used"] = len(initializer_jobs)

            assignment_sorted = {k: [] for k in initializer_jobs}

            for i in I:
                for k in range(i + 1):
                    if (i, k) in Delta_red and x[i, k].X > 0.5:
                        assignment_sorted.setdefault(k, []).append(i)
                        break

            assignment_original = {}

            for k, assigned_jobs in assignment_sorted.items():
                original_initializer = original_ids[k]
                original_assigned_jobs = [original_ids[i] for i in assigned_jobs]
                assignment_original[original_initializer] = original_assigned_jobs

            result["assignment"] = assignment_original

    return result


if __name__ == "__main__":
    jobs = [
        (0, 4, 40),
        (1, 5, 30),
        (4, 7, 50),
        (6, 9, 40),
    ]

    C = 100

    result = solve_model3_optimized(
        jobs=jobs,
        C=C,
        gamma=1.0,
        time_limit=1800,
        verbose=True,
        use_lifting=True,
        use_material_lb=True
    )

    print("\n--- RISULTATO MODELLO 3 OTTIMIZZATO ---")
    print("Status:", result["status"])
    print("Objective:", result["objective"])
    print("Server usati:", result["servers_used"])
    print("Fire-up:", result["fireups"])
    print("Assignment:", result["assignment"])
    print("h0:", result["h0"])
    print("Variabili:", result["n_vars"])
    print("Vincoli:", result["n_constrs"])
    print("Nonzero:", result["n_nonzeros"])
