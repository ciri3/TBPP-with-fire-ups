import gurobipy as gp
from gurobipy import GRB
import math


def solve_model1_optimized(
    jobs,
    C=100,
    gamma=1.0,
    time_limit=1800,
    verbose=True
):
    """
    Modello 1 ottimizzato per TBPP-FU.

    Include:
    - Lit-A: lower bound h0 sui server usati
    - Lit-B: ordinamento dei server z[k] >= z[k+1]
    - w binarie
    - R1: struttura triangolare x[i,k] solo per k <= i
    - R2(a): z[k] <= sum_t w[t,k]

    jobs: lista di tuple (s_i, e_i, c_i)
          con indici Python 0-based.
    C: capacità server.
    gamma: peso dei fire-up.
    """

    # =========================
    # 1. PREPROCESSING DATI
    # =========================

    n = len(jobs)

    I = range(n)
    K = range(n)

    s = {i: jobs[i][0] for i in I}
    e = {i: jobs[i][1] for i in I}
    c = {i: jobs[i][2] for i in I}

    # Controllo di fattibilità base
    for i in I:
        if c[i] > C:
            raise ValueError(f"Job {i} ha c_i={c[i]} > C={C}. Istanza infeasible.")

    # Insieme globale dei tempi
    #T_global = sorted(set([s[i] for i in I] + [e[i] for i in I]))

    # Insieme globale degli starting times
    TS_global = sorted(set(s[i] for i in I))

    # =========================
    # 2. R1: STRUTTURA TRIANGOLARE
    # =========================
    # Delta = {(i,k) | k <= i}
    # Attenzione: con indici Python 0-based, k <= i.

    Delta = [(i, k) for i in I for k in K if k <= i]

    # Per ogni server k, posso assegnare solo job i >= k.
    jobs_allowed_on_server = {
        k: [i for i in I if i >= k]
        for k in K
    }

    # T(k) = unione di start/end dei job i >= k
    T_k = {}
    TS_k = {}

    for k in K:
        times_k = set()
        start_times_k = set()

        for i in jobs_allowed_on_server[k]:
            times_k.add(s[i])
            times_k.add(e[i])
            start_times_k.add(s[i])

        T_k[k] = sorted(times_k)
        TS_k[k] = sorted(start_times_k)

    # Predecessore pred_k(t) dentro T(k)
    pred_k = {}

    for k in K:
        pred_k[k] = {}
        times = T_k[k]

        for idx, t in enumerate(times):
            if idx == 0:
                pred_k[k][t] = None
            else:
                pred_k[k][t] = times[idx - 1]

    # Parametro a_it: job i attivo al tempo t?
    # Lo calcoliamo solo quando serve. a(i,t)
    def is_active(i, t):
        return 1 if s[i] <= t < e[i] else 0

    # =========================
    # 3. Lit-A: MATERIAL BOUND h0
    # =========================
    # h0 = ceil(max_t sum_i a_it c_i / C)
    # Basta considerare t in TS_global.

    max_load = 0

    for t in TS_global:
        load_t = sum(c[i] for i in I if is_active(i, t))
        max_load = max(max_load, load_t)

    h0 = math.ceil(max_load / C)

    if verbose:
        print(f"Material bound h0 = {h0}")

    # =========================
    # 4. MODELLO GUROBI
    # =========================

    model = gp.Model("Model1_Optimized_TBPP_FU")

    if not verbose:
        model.Params.OutputFlag = 0

    model.Params.TimeLimit = time_limit

    # =========================
    # 5. VARIABILI
    # =========================

    # x[i,k] solo per k <= i
    x = model.addVars(
        Delta,
        vtype=GRB.BINARY,
        name="x"
    )

    # y[t,k] solo per t in T(k)
    y_index = [
        (t, k)
        for k in K
        for t in T_k[k]
    ]

    y = model.addVars(
        y_index,
        vtype=GRB.BINARY,
        name="y"
    )

    # w[t,k] solo per t in TS(k), binarie
    w_index = [
        (t, k)
        for k in K
        for t in TS_k[k]
    ]

    w = model.addVars(
        w_index,
        vtype=GRB.BINARY,
        name="w"
    )

    # z[k]
    z = model.addVars(
        K,
        vtype=GRB.BINARY,
        name="z"
    )

    # =========================
    # 6. OBIETTIVO
    # =========================

    model.setObjective(
        gamma * gp.quicksum(w[t, k] for (t, k) in w_index)
        + gp.quicksum(z[k] for k in K),
        GRB.MINIMIZE
    )

    # =========================
    # 7. VINCOLI BASE DEL MODELLO 1
    # =========================

    # (1) y_tk <= carico_tk <= C y_tk
    for k in K:
        for t in T_k[k]:

            load_expr = gp.quicksum(
                c[i] * is_active(i, t) * x[i, k]
                for i in jobs_allowed_on_server[k]
                if (i, k) in x
            )

            # Se y[t,k] = 1, il server deve avere carico positivo.
            # Se il carico è positivo, y[t,k] deve essere 1.
            model.addConstr(
                y[t, k] <= load_expr,
                name=f"active_if_positive_load_k{k}_t{t}"
            )

            model.addConstr(
                load_expr <= C * y[t, k],
                name=f"capacity_k{k}_t{t}"
            )

    # (2) Ogni job assegnato esattamente una volta
    for i in I:
        model.addConstr(
            gp.quicksum(x[i, k] for k in K if (i, k) in x) == 1,
            name=f"assign_job_{i}"
        )

    # (3) Se job i è su server k, allora server k è attivo a s_i
    for (i, k) in Delta:
        model.addConstr(
            x[i, k] <= y[s[i], k],
            name=f"x_implies_y_start_i{i}_k{k}"
        )

    # (4) Se server k è attivo in t, allora server k è usato
    for k in K:
        for t in T_k[k]:
            model.addConstr(
                y[t, k] <= z[k],
                name=f"y_implies_z_k{k}_t{t}"
            )

    # (5) Fire-up: y[t,k] - y[pred,t,k] <= w[t,k]
    for k in K:
        for t in TS_k[k]:

            pred = pred_k[k][t]

            if pred is None:
                # Prima attivazione possibile: y[t,k] <= w[t,k]
                model.addConstr(
                    y[t, k] <= w[t, k],
                    name=f"fireup_first_k{k}_t{t}"
                )
            else:
                model.addConstr(
                    y[t, k] - y[pred, k] <= w[t, k],
                    name=f"fireup_k{k}_t{t}"
                )

    # =========================
    # 8. Lit-A: FISSARE I PRIMI h0 SERVER
    # =========================

    for k in range(h0):
        model.addConstr(
            z[k] == 1,
            name=f"fix_used_server_{k}"
        )

    # =========================
    # 9. Lit-B: ORDINAMENTO SERVER
    # =========================
    # z[k] >= z[k+1] solo da k = h0 in poi.
    # Con indici 0-based: k va da h0 a n-2.

    for k in range(h0, n - 1):
        model.addConstr(
            z[k] >= z[k + 1],
            name=f"server_order_{k}"
        )

    # =========================
    # 10. R2(a): SERVER USATO => ALMENO UN FIRE-UP
    # =========================

    for k in K:
        model.addConstr(
            z[k] <= gp.quicksum(w[t, k] for t in TS_k[k]),
            name=f"used_server_has_fireup_{k}"
        )

    # =========================
    # 11. OTTIMIZZAZIONE
    # =========================

    model.optimize()

    # =========================
    # 12. ESTRAZIONE RISULTATI
    # =========================

    result = {
        "status": model.Status,
        "objective": None,
        "servers_used": None,
        "fireups": None,
        "assignments": {},
        "h0": h0,
        "runtime": model.Runtime,
        "gap": None,
        "model": model,
    }

    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        if model.SolCount > 0:
            result["objective"] = model.ObjVal

            result["servers_used"] = sum(
                1 for k in K if z[k].X > 0.5
            )

            result["fireups"] = sum(
                1 for (t, k) in w_index if w[t, k].X > 0.5
            )

            for i in I:
                for k in K:
                    if (i, k) in x and x[i, k].X > 0.5:
                        result["assignments"][i] = k

            if model.Status != GRB.OPTIMAL:
                result["gap"] = model.MIPGap

    if verbose:
        print("\n========== RISULTATI MODELLO 1 OTTIMIZZATO ==========")
        print(f"Status Gurobi: {model.Status}")

        if result["objective"] is not None:
            print(f"Objective value: {result['objective']}")
            print(f"Server usati: {result['servers_used']}")
            print(f"Fire-ups: {result['fireups']}")
            print(f"Runtime: {result['runtime']:.4f} s")

            if result["gap"] is not None:
                print(f"MIP gap: {result['gap']:.6f}")

            print("\nAssegnamenti job -> server:")
            for i, k in result["assignments"].items():
                print(f"Job {i} -> Server {k}")
        else:
            print("Nessuna soluzione trovata.")

    return result