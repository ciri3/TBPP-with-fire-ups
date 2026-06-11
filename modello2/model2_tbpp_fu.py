import gurobipy as gp
from gurobipy import GRB


def solve_model2(jobs, C, gamma=1.0, time_limit=None, verbose=True, binary_w=True):
    """
    Model 2 per il Temporal Bin Packing Problem with Fire-Ups.

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

    # --------------------------------------------------
    # 1. Ordinamento dei job per tempo di inizio crescente
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

    # Insieme dei tempi di inizio distinti
    TS = sorted(set(s))

    # --------------------------------------------------
    # 2. Costruzione degli insiemi delta_i e delta_plus_i
    # --------------------------------------------------

    #insiemi di insiemi
    delta = {}
    delta_plus = {}

    for i in I:
        delta[i] = []
        delta_plus[i] = []

        for j in range(i):
            # delta_i = job precedenti ancora attivi quando parte i
            if s[i] < e[j]:
                delta[i].append(j)

            # delta_plus_i = job precedenti attivi oppure appena terminati
            if s[i] <= e[j]:
                delta_plus[i].append(j)

    # --------------------------------------------------
    # 3. Creazione modello Gurobi
    # --------------------------------------------------
    model = gp.Model("TBPP_FU_Model2")

    if not verbose:
        model.Params.OutputFlag = 0

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # --------------------------------------------------
    # 4. Variabili decisionali
    # --------------------------------------------------

    # x[i,k] = 1 se il job i è assegnato al server k
    x = model.addVars(I, K, vtype=GRB.BINARY, name="x")

    # z[k] = 1 se il server k è usato
    z = model.addVars(K, vtype=GRB.BINARY, name="z")

    # w[t,k] = 1 se il server k fa fire-up al tempo t
    if binary_w:
        w = model.addVars(TS, K, vtype=GRB.BINARY, name="w")
    else:
        w = model.addVars(TS, K, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

    # --------------------------------------------------
    # 5. Funzione obiettivo
    # --------------------------------------------------
    model.setObjective(
        gamma * gp.quicksum(w[t, k] for t in TS for k in K)
        + gp.quicksum(z[k] for k in K),
        GRB.MINIMIZE
    )

    # --------------------------------------------------
    # 6. Vincoli
    # --------------------------------------------------

    # (10) Vincoli di capacità
    for i in I:
        for k in K:
            model.addConstr(
                gp.quicksum(c[j] * x[j, k] for j in delta[i])
                + c[i] * x[i, k]
                <= C * z[k],
                name=f"capacity_i{i}_k{k}"
            )

    # (11) Ogni job assegnato esattamente a un server
    for i in I:
        model.addConstr(
            gp.quicksum(x[i, k] for k in K) == 1,
            name=f"assign_i{i}"
        )

    # (12) Se assegno un job al server k, allora il server k è usato
    for i in I:
        for k in K:
            model.addConstr(
                x[i, k] <= z[k],
                name=f"link_x_z_i{i}_k{k}"
            )

    # (13) Vincoli per i fire-up
    for i in I:
        start_i = s[i]

        for k in K:
            model.addConstr(
                gp.quicksum(x[j, k] for j in delta_plus[i])
                - x[i, k]
                + w[start_i, k]
                >= 0,
                name=f"fireup_i{i}_k{k}"
            )

    # --------------------------------------------------
    # 7. Ottimizzazione
    # --------------------------------------------------
    model.optimize()

    # --------------------------------------------------
    # 8. Estrazione soluzione
    # --------------------------------------------------
    result = {
        "status": model.Status,
        "objective": None,
        "servers_used": None,
        "runtime": model.Runtime, # necessario per la scalabilità
        "fireups": None,
        "assignment": {},
        "fireup_times": {},
        "model": model,
        "num_vars": model.NumVars,
        "num_constrs": model.NumConstrs,
    }

    if model.SolCount == 0:
        return result

    result["objective"] = model.ObjVal

    used_servers = [k for k in K if z[k].X > 0.5]
    result["servers_used"] = len(used_servers)

    fireup_list = []
    for k in K:
        for t in TS:
            if w[t, k].X > 0.5:
                fireup_list.append((k, t))

    result["fireups"] = len(fireup_list)

    # Assegnamento job -> server
    assignment = {}
    for i in I:
        for k in K:
            if x[i, k].X > 0.5:
                original_job_id = original_ids[i]
                assignment[original_job_id] = k
                break

    result["assignment"] = assignment

    # Fire-up per server
    fireup_times = {}
    for k, t in fireup_list:
        fireup_times.setdefault(k, []).append(t)

    result["fireup_times"] = fireup_times

    return result


if __name__ == "__main__":

    # Esempio piccolo
    jobs = [
        (1, 2, 1),
        (1, 4, 2),
        (3, 4, 1),
    ]

    C = 2
    gamma = 1.0

    result = solve_model2(
        jobs=jobs,
        C=C,
        gamma=gamma,
        time_limit=1800,
        verbose=True,
        binary_w=True
    )

    print("\n--- RISULTATI MODELLO 2 ---")
    print("Status Gurobi:", result["status"])
    print("Objective:", result["objective"])
    print("Server usati:", result["servers_used"])
    print("Fire-up:", result["fireups"])
    print("Assignment job -> server:", result["assignment"])
    print("Fire-up times:", result["fireup_times"])