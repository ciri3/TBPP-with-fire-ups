import gurobipy as gp
from gurobipy import GRB
import math


def solve_model2_optimized(jobs, C, gamma=1.0, time_limit=None, verbose=True):
    """
    Modello 2 ottimizzato per TBPP-FU.

    Include:
    - R0 / Lit-A: lower bound h0 sui server usati
    - R0 / Lit-B: ordinamento z[k] >= z[k+1]
    - w binarie
    - R1: variabili x triangolari, solo k <= i
    - R1: w solo su TS(k)
    - R2: z[k] <= sum_t w[t,k]
    - R4: delta_plus senza job con stesso start time
    - R5: vincoli capacità solo per starting time non dominati
    """

    # --------------------------------------------------
    # 1. Ordinamento job per start time crescente
    # --------------------------------------------------
    indexed_jobs = list(enumerate(jobs))
    indexed_jobs.sort(key=lambda item: (item[1][0], item[1][1], item[0]))

    # original_ids serve per tornare agli indici originali dei job
    original_ids = [idx for idx, job in indexed_jobs]

    s = [job[0] for idx, job in indexed_jobs]
    e = [job[1] for idx, job in indexed_jobs]
    c = [job[2] for idx, job in indexed_jobs]

    n = len(jobs)
    I = range(n)
    K = range(n)

    for i in I:
        if c[i] > C:
            raise ValueError(f"Job {i} ha capacità {c[i]} > C={C}")

    # --------------------------------------------------
    # 2. Insiemi temporali
    # --------------------------------------------------
    T = sorted(set(s) | set(e))
    TS = sorted(set(s))
    E_times = set(e)

    # --------------------------------------------------
    # 3. Lower bound h0, Lit-A
    # --------------------------------------------------
    max_load = 0
    for t in TS:
        load_t = sum(c[i] for i in I if s[i] <= t < e[i])
        max_load = max(max_load, load_t)

    h0 = math.ceil(max_load / C)

    # --------------------------------------------------
    # 4. R1: struttura triangolare DeltaGrande
    # --------------------------------------------------
    DeltaGrande = [(i, k) for i in I for k in K if k <= i]

    jobs_allowed_on_server = {
        k: [i for i in I if i >= k]
        for k in K
    }

    # TS(k) = starting times dei job che possono andare su server k
    TS_k = {
        k: sorted(set(s[i] for i in jobs_allowed_on_server[k]))
        for k in K
    }

    W_index = [
        (t, k)
        for k in K
        for t in TS_k[k]
    ]

    # --------------------------------------------------
    # 5. DeltaGrande e delta_plus
    # --------------------------------------------------
    delta = {}
    delta_plus = {}

    for i in I:
        delta[i] = []
        delta_plus[i] = []

        for j in range(i):
            # delta_i: job precedenti attivi quando parte i
            if s[i] < e[j]:
                delta[i].append(j)

            # R4: delta_plus_i senza job con stesso start time
            if s[i] <= e[j] and s[i] != s[j]:
                delta_plus[i].append(j)

    # --------------------------------------------------
    # 6. R5: starting time non dominati
    # --------------------------------------------------
    succ_T = {}
    for idx, t in enumerate(T[:-1]):
        succ_T[t] = T[idx + 1]

    TS_nd = {
        t for t in TS
        if succ_T.get(t) in E_times
    }

    # Nota: se l'ultimo tempo fosse uno start senza successore, lo teniamo
    # per sicurezza, anche se in istanze normali ogni job ha e_i > s_i.
    for t in TS:
        if t not in succ_T:
            TS_nd.add(t)

    # --------------------------------------------------
    # 7. Creazione modello
    # --------------------------------------------------
    model = gp.Model("TBPP_FU_Model2_Optimized")

    if not verbose:
        model.Params.OutputFlag = 0

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # --------------------------------------------------
    # 8. Variabili decisionali
    # --------------------------------------------------

    # R1: x solo per k <= i
    x = model.addVars(DeltaGrande, vtype=GRB.BINARY, name="x")

    z = model.addVars(K, vtype=GRB.BINARY, name="z")

    # R0: w binarie e solo su TS(k)
    w = model.addVars(W_index, vtype=GRB.BINARY, name="w")

    # --------------------------------------------------
    # 9. Funzione obiettivo
    # --------------------------------------------------
    model.setObjective(
        gamma * gp.quicksum(w[t, k] for (t, k) in W_index)
        + gp.quicksum(z[k] for k in K),
        GRB.MINIMIZE
    )

    # --------------------------------------------------
    # 10. Vincoli
    # --------------------------------------------------

    # Lit-A: primi h0 server fissati a 1
    for k in range(h0):
        model.addConstr(z[k] == 1, name=f"litA_fix_z_{k}")

    # Lit-B: ordinamento server, solo dopo h0
    for k in range(h0, n - 1):
        model.addConstr(z[k] >= z[k + 1], name=f"litB_order_z_{k}")

    # (10) R5: vincoli capacità solo per start time non dominati
    for i in I:
        if s[i] not in TS_nd:
            continue

        for k in K:

            #equivalente a no in DeltaGrande ma piu robusto
            if (i, k) not in x:
                continue

            model.addConstr(
                gp.quicksum(
                    c[j] * x[j, k]
                    for j in delta[i]
                    if (j, k) in x
                )
                + c[i] * x[i, k]
                <= C * z[k],
                name=f"capacity_i{i}_k{k}"
            )

    # (11) ogni job assegnato esattamente a un server ammesso
    for i in I:
        model.addConstr(
            gp.quicksum(x[i, k] for k in K if (i, k) in x) == 1,
            name=f"assign_i{i}"
        )

    # (12) link x-z
    for (i, k) in DeltaGrande:
        model.addConstr(
            x[i, k] <= z[k],
            name=f"link_x_z_i{i}_k{k}"
        )

    # (13) vincoli fire-up con delta_plus modificato da R4
    for i in I:
        start_i = s[i]

        for k in K:
            if (i, k) not in x:
                continue

            if (start_i, k) not in w:
                continue

            model.addConstr(
                gp.quicksum(
                    x[j, k]
                    for j in delta_plus[i]
                    if (j, k) in x
                )
                - x[i, k]
                + w[start_i, k]
                >= 0,
                name=f"fireup_i{i}_k{k}"
            )

    # R2: se server k è usato, deve avere almeno un fire-up
    for k in K:
        model.addConstr(
            z[k] <= gp.quicksum(w[t, k] for t in TS_k[k]),
            name=f"R2_z_le_fireups_k{k}"
        )

    # --------------------------------------------------
    # 11. Ottimizzazione
    # --------------------------------------------------
    model.optimize()

    # --------------------------------------------------
    # 12. Estrazione soluzione
    # --------------------------------------------------
    result = {
        "status": model.Status,
        "objective": None,
        "servers_used": None,
        "runtime": model.Runtime,
        "fireups": None,
        "assignment": {},
        "fireup_times": {},
        "h0": h0,
        "n_variables": model.NumVars,
        "n_constraints": model.NumConstrs,
        "nnz": model.NumNZs,
        "model": model,
    }

    if model.SolCount == 0:
        return result

    result["objective"] = model.ObjVal

    used_servers = [k for k in K if z[k].X > 0.5]
    result["servers_used"] = len(used_servers)

    fireup_list = []
    for (t, k) in W_index:
        if w[t, k].X > 0.5:
            fireup_list.append((k, t))

    result["fireups"] = len(fireup_list)

    assignment = {}
    for i in I:
        for k in K:
            if (i, k) in x and x[i, k].X > 0.5:
                original_job_id = original_ids[i]
                assignment[original_job_id] = k
                break

    result["assignment"] = assignment

    fireup_times = {}
    for k, t in fireup_list:
        fireup_times.setdefault(k, []).append(t)

    result["fireup_times"] = fireup_times

    return result