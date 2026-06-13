# modello3/model3_optimized_no_lifting_tbpp_fu.py

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


def solve_model3_optimized(
    jobs,
    C,
    gamma=1.0,
    time_limit=None,
    verbose=True
):
    """
    Include:
    - R0: lower bound h0 sul numero di server inizializzatori
    - R1*: Delta_red, cioè eliminazione delle coppie (i,k) impossibili
    - R2*: disuguaglianza valida x[k,k] <= w[k]
    - R5*: Delta_red_nd, cioè eliminazione dei vincoli di capacità dominati

    R4 NON viene applicata nel Modello 3:
    escludere i job con stesso start time da delta_plus_i porterebbe
    a contare fire-up doppi in alcuni casi.
    """

    # ------------------------------------------------------------
    # Ordinamento dei job
    # ------------------------------------------------------------
    indexed_jobs = list(enumerate(jobs))
    indexed_jobs.sort(key=lambda item: (item[1][0], item[1][1], item[0]))

    original_ids = [idx for idx, job in indexed_jobs]
    sorted_jobs = [job for idx, job in indexed_jobs]

    n = len(sorted_jobs)
    I = range(n)

    # Controllo fattibilità elementare
    for i, (_, _, c_i) in enumerate(sorted_jobs):
        if c_i > C:
            raise ValueError(
                f"Job {i} ha richiesta c_i={c_i} maggiore della capacità C={C}."
            )

    # ------------------------------------------------------------
    # R0: lower bound h0 sul numero minimo di server
    # ------------------------------------------------------------
    
    start_times = sorted({job[0] for job in sorted_jobs})

    max_load = 0
    for t in start_times:
        load_t = sum(
            c_i
            for s_i, e_i, c_i in sorted_jobs
            if s_i <= t < e_i
        )
        max_load = max(max_load, load_t)

    h0 = math.ceil(max_load / C)

    # ------------------------------------------------------------
    # Costruzione di delta_i e delta_plus_i
    # ------------------------------------------------------------
    # delta[i] contiene i job j < i attivi allo start time di i.
    # delta_plus[i] contiene i job j < i ancora attivi oppure appena terminati
    # allo start time di i.
    delta = {}
    delta_plus = {}

    for i in I:
        s_i = sorted_jobs[i][0]
        

        delta[i] = []
        delta_plus[i] = []

        for j in range(i):
            e_j = sorted_jobs[j][1]

            if s_i < e_j:
                delta[i].append(j)

            if s_i <= e_j:
                delta_plus[i].append(j)

    # ------------------------------------------------------------
    # R1*: costruzione di Delta_red
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

            job_i = sorted_jobs[i]
            job_k = sorted_jobs[k]

            c_i = sorted_jobs[i][2]
            c_k = sorted_jobs[k][2]

            if intervals_overlap(job_i, job_k) and c_i + c_k > C:
                continue

            Delta_red.add((i, k))

    # ------------------------------------------------------------
    # R5*: costruzione di Delta_red_nd
    # ------------------------------------------------------------
    # Se i e j hanno stesso start time, k < i < j,
    # (i,k), (j,k) in Delta_red, allora (j,k) domina (i,k).
    # Questa riduzione vale SOLO per i vincoli di capacità.
    Delta_red_nd = set()

    for (i, k) in Delta_red:

        #scartiamo la diagonale perchè il job inizializzatore non è coinvolto nei vincoli di capacità non potendo superare la capacità del server da solo
        if i == k:
            continue

        s_i = sorted_jobs[i][0]
        dominated = False

        for j in range(i + 1, n):
            s_j = sorted_jobs[j][0]

            if s_j > s_i:
                break

            if s_j == s_i and (j, k) in Delta_red:
                dominated = True
                break

        if not dominated:
            Delta_red_nd.add((i, k))

    # ------------------------------------------------------------
    # Creazione modello Gurobi
    # ------------------------------------------------------------
    model = gp.Model("TBPP_FU_Model3_Optimized_NoLifting")

    if not verbose:
        model.setParam("OutputFlag", 0)

    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)

    # ------------------------------------------------------------
    # Variabili
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
    # Funzione obiettivo
    # ------------------------------------------------------------
    model.setObjective(
        gamma * gp.quicksum(w[i] for i in I)
        + gp.quicksum(x[k, k] for k in I),
        GRB.MINIMIZE
    )

    # ------------------------------------------------------------
    # R0: lower bound sui server inizializzatori
    # ------------------------------------------------------------
    # Nel Modello 3 non abbiamo z[k].
    # Il server usato/iniziato da k è rappresentato da x[k,k].
    
    model.addConstr(
        gp.quicksum(x[k, k] for k in I) >= h0,
        name="material_lower_bound"
    )

    # ------------------------------------------------------------
    # Vincoli di assegnamento 
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
    # Vincoli di capacità ridotti 
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
    # Vincoli di collegamento 
    # ------------------------------------------------------------
    for (i, k) in sorted(Delta_red):
        if i != k:
            model.addConstr(
                x[i, k] <= x[k, k],
                name=f"link_{i}_{k}"
            )

    # ------------------------------------------------------------
    # R2*: disuguaglianze valide x[k,k] <= w[k]
    # ------------------------------------------------------------
    # Se k inizializza un server, allora k causa sicuramente un fire-up.
    for k in I:
        model.addConstr(
            x[k, k] <= w[k],
            name=f"valid_initializer_fireup_{k}"
        )

    # ------------------------------------------------------------
    # Vincoli sui fire-up 
    # ------------------------------------------------------------
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
    # Risoluzione
    # ------------------------------------------------------------
    model.optimize()

    # ------------------------------------------------------------
    # Estrazione soluzione
    # ------------------------------------------------------------
    result = {
        "status": model.Status,
        "objective": None,
        "fireups": None,
        "servers_used": None,
        "assignment": None,
        "sorted_jobs": sorted_jobs,
        "original_ids": original_ids,
        "runtime": model.Runtime,
        "h0": h0,
        "n_vars": model.NumVars,
        "n_constrs": model.NumConstrs,
        "n_nonzeros": model.NumNZs,
        "Delta_red_size": len(Delta_red),
        "Delta_red_nd_size": len(Delta_red_nd),
        "model": model,
        "num_vars": model.NumVars,
        "num_constrs": model.NumConstrs,
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

"""
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
    )

    print("\n--- RISULTATO MODELLO 3 OTTIMIZZATO SENZA LIFTING ---")
    print("Status:", result["status"])
    print("Objective:", result["objective"])
    print("Server usati:", result["servers_used"])
    print("Fire-up:", result["fireups"])
    print("Assignment:", result["assignment"])
    print("h0:", result["h0"])
    print("Variabili:", result["n_vars"])
    print("Vincoli:", result["n_constrs"])
    print("Nonzero:", result["n_nonzeros"])"""
