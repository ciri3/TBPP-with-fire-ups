import gurobipy as gp
from gurobipy import GRB


def solve_model1(jobs, C, gamma=1.0, time_limit=None, verbose=True, binary_w=True):
    """
    jobs: lista di tuple (s_i, e_i, c_i)
          s_i = start time
          e_i = end time
          c_i = richiesta di capacità

    C: capacità massima di ogni server
    gamma: peso dei fire-ups nella funzione obiettivo
    time_limit: limite di tempo in secondi
    verbose: se True stampa log Gurobi
    binary_w: se True usa w binarie; se False usa w continue >= 0
    """

    # -----------------------------
    # Preprocessing dati
    # -----------------------------

    n = len(jobs)

    I = range(n)
    K = range(n)

    s = [jobs[i][0] for i in I]
    e = [jobs[i][1] for i in I]
    c = [jobs[i][2] for i in I]

    for i in I:
        if c[i] > C:
            raise ValueError(f"Job {i} ha c_i={c[i]} > C={C}. Istanza infeasible.")

    # Insieme dei tempi rilevanti:
    # tutti gli start time e tutti gli end time
    T = sorted(set(s + e))

    # Insieme degli start time
    TS = sorted(set(s))

    # Predecessore temporale nella lista ordinata T
    pred = {}
    for idx, t in enumerate(T):
        if idx == 0:
            pred[t] = None
        else:
            pred[t] = T[idx - 1]

    # Parametro a_it:
    # a[i,t] = 1 se job i è attivo al tempo t
    a = {}
    for i in I:
        for t in T:
            a[i, t] = 1 if s[i] <= t < e[i] else 0

    # -----------------------------
    # Creazione modello
    # -----------------------------

    model = gp.Model("TBPP_FU_Model1")

    if not verbose:
        model.Params.OutputFlag = 0

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # -----------------------------
    # Variabili
    # -----------------------------

    # x[i,k] = 1 se job i assegnato al server k
    x = model.addVars(I, K, vtype=GRB.BINARY, name="x")

    # y[t,k] = 1 se server k attivo al tempo t
    y = model.addVars(T, K, vtype=GRB.BINARY, name="y")

    # z[k] = 1 se server k usato
    z = model.addVars(K, vtype=GRB.BINARY, name="z")

    # w[t,k] = 1 se server k ha un fire-up al tempo t
    if binary_w:
        w = model.addVars(TS, K, vtype=GRB.BINARY, name="w")
    else:
        w = model.addVars(TS, K, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

    # -----------------------------
    # Funzione obiettivo
    # -----------------------------
    
    #in Ts
    model.setObjective(
        gamma * gp.quicksum(w[t, k] for t in TS for k in K)
        + gp.quicksum(z[k] for k in K),
        GRB.MINIMIZE
    )

    # ----------------------------------
    # Vincoli di capacità e attività
    # ----------------------------------
    #
    # y_tk <= sum_i c_i a_it x_ik <= C y_tk

    for t in T:
        for k in K:
            load_tk = gp.quicksum(c[i] * a[i, t] * x[i, k] for i in I)

            # Parte sinistra:
            # se y[t,k] = 1, allora deve esserci carico positivo
            model.addConstr(
                y[t, k] <= load_tk,
                name=f"activity_lb_t{t}_k{k}"
            )

            # Parte destra:
            # capacità del server
            model.addConstr(
                load_tk <= C * y[t, k],
                name=f"capacity_t{t}_k{k}"
            )

    # -----------------------------
    # Ogni job assegnato esattamente una volta
    # -----------------------------

    for i in I:
        model.addConstr(
            gp.quicksum(x[i, k] for k in K) == 1,
            name=f"assign_job_{i}"
        )

    # -----------------------------
    # Se assegno job i a server k, allora il server k deve essere attivo allo start time s_i
    # -----------------------------

    for i in I:
        for k in K:
            model.addConstr(
                x[i, k] <= y[s[i], k],
                name=f"x_y_link_i{i}_k{k}"
            )

    # -----------------------------
    # Se un server è attivo, allora è usato
    # -----------------------------

    for t in T:
        for k in K:
            model.addConstr(
                y[t, k] <= z[k],
                name=f"y_z_link_t{t}_k{k}"
            )

    # -----------------------------
    # Conteggio dei fire-ups
    # -----------------------------
    #
    # y[t,k] - y[pred(t),k] <= w[t,k]
    #
    # Se t è il primo tempo, assumiamo y[pred(t),k] = 0.

    for t in TS:
        for k in K:
            previous_t = pred[t]

            if previous_t is None:
                model.addConstr(
                    y[t, k] <= w[t, k],
                    name=f"fireup_first_t{t}_k{k}"
                )
            else:
                model.addConstr(
                    y[t, k] - y[previous_t, k] <= w[t, k],
                    name=f"fireup_t{t}_k{k}"
                )

    # -----------------------------
    # Ottimizzazione
    # -----------------------------

    model.optimize()

    # -----------------------------
    # Estrazione risultati
    # -----------------------------

    #if status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    #    return {
    #        "status": status,
    #        "message": "Il modello non ha trovato #una soluzione utilizzabile.",
    #        "model": model
    #    }
    
    status = model.Status

    if model.SolCount == 0:
        return {
            "status": status,
            "message": "Il modello non ha trovato nessuna soluzione ammissibile.",
            "model": model
        }

    assignments = {}

    for i in I:
        for k in K:
            if x[i, k].X > 0.5:
                assignments[i] = k

    used_servers = [k for k in K if z[k].X > 0.5]

    fireups = []
    for t in TS:
        for k in K:
            if w[t, k].X > 0.5:
                fireups.append((t, k))

    server_jobs = {k: [] for k in used_servers}

    for i, k in assignments.items():
        server_jobs[k].append(i)

    result = {
        "status": status,
        "objective": model.ObjVal if model.SolCount > 0 else None,
        "bound": model.ObjBound if model.SolCount > 0 else None,
        "gap": model.MIPGap if model.SolCount > 0 else None,
        "runtime": model.Runtime,
        "num_vars": model.NumVars,
        "num_constraints": model.NumConstrs,
        "num_nonzeros": model.NumNZs,
        "servers_used": len(used_servers),
        "num_fireups": len(fireups),
        "used_servers": used_servers,
        "fireups": fireups,
        "assignments": assignments,
        "server_jobs": server_jobs,
        "T": T,
        "TS": TS,
        "model": model,
        "num_vars": model.NumVars,
        "num_constrs": model.NumConstrs
    }

    return result

"""
if __name__ == "__main__":

    # Esempio piccolo
    jobs = [
        # (start, end, size)
        (1, 3, 2),   # job 0
        (1, 2, 3),   # job 1
        (3, 4, 1),   # job 2
        (3, 4, 3),   # job 3
    ]

    C = 3
    gamma = 1.0

    result = solve_model1(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=1800,
        verbose=True,
        binary_w=True
    )

    print("\n---------------- RISULTATI MODEL 1 ----------------")
    print("Status:", result["status"])
    print("Objective:", result["objective"])
    print("Bound:", result["bound"])
    print("Gap:", result["gap"])
    print("Runtime:", result["runtime"])
    print("Numero variabili:", result["num_vars"])
    print("Numero vincoli:", result["num_constraints"])
    print("Numero nonzeri:", result["num_nonzeros"])
    print("Server usati:", result["servers_used"])
    print("Fire-ups:", result["num_fireups"])
    print("Lista fire-ups:", result["fireups"])
    print("Assegnamenti job -> server:", result["assignments"])
    print("Job per server:", result["server_jobs"])"""