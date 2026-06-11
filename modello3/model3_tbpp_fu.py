# modello3/model3_tbpp_fu.py

import gurobipy as gp
from gurobipy import GRB


def intervals_overlap(job_a, job_b):
    """
    Controlla se due job si sovrappongono temporalmente.

    job = (s, e, c)

    Intervalli considerati: [s, e)
    Quindi due job si sovrappongono se:
        s_a < e_b e s_b < e_a
    """
    s_a, e_a, _ = job_a
    s_b, e_b, _ = job_b

    return s_a < e_b and s_b < e_a


def solve_model3(jobs, C, gamma=1.0, time_limit=None, verbose=True):
    """
    Model 3 per il Temporal Bin Packing Problem with Fire-Ups.

    jobs: lista di tuple (s_i, e_i, c_i)
          s_i = start time
          e_i = end time
          c_i = richiesta di capacità

    C: capacità massima di ogni server
    gamma: peso dei fire-ups nella funzione obiettivo
    time_limit: limite di tempo in secondi
    verbose: se True stampa log Gurobi

    Ritorna un dizionario con:
        - status
        - objective
        - fireups
        - servers_used
        - assignment
        - sorted_jobs
        - original_ids
    """

    # ------------------------------------------------------------
    # 1. Ordinamento dei job
    # ------------------------------------------------------------
    # Nel Modello 3 è fondamentale ordinare i job per start time.
    # Usiamo anche end time e indice originale per rompere i pareggi.
    indexed_jobs = list(enumerate(jobs))
    indexed_jobs.sort(key=lambda item: (item[1][0], item[1][1], item[0]))

    original_ids = [idx for idx, job in indexed_jobs]
    sorted_jobs = [job for idx, job in indexed_jobs]

    n = len(sorted_jobs)
    I = range(n)

    # ------------------------------------------------------------
    # 2. Costruzione degli insiemi delta_i e delta_plus_i
    # ------------------------------------------------------------
    # delta[i] contiene i job j < i ancora attivi quando inizia i:
    #     s_i < e_j
    #
    # delta_plus[i] contiene i job j < i attivi o appena terminati:
    #     s_i <= e_j
    #
    # delta serve per la capacità.
    # delta_plus serve per i fire-up.

    delta = {}
    delta_plus = {}

    for i in I:
        s_i, e_i, c_i = sorted_jobs[i]

        delta[i] = []
        delta_plus[i] = []

        for j in range(i):
            s_j, e_j, c_j = sorted_jobs[j]

            if s_i < e_j:
                delta[i].append(j)

            if s_i <= e_j:
                delta_plus[i].append(j)

    # ------------------------------------------------------------
    # 3. Costruzione di Delta_red
    # ------------------------------------------------------------
    # Delta base:
    #     (i, k) con k <= i
    #
    # Delta_red elimina coppie impossibili.
    #
    # Se i e k si sovrappongono temporalmente e c_i + c_k > C,
    # allora non possono stare sullo stesso server.
    # Quindi x[i,k] sarebbe sempre 0 e possiamo non crearla.

    Delta_red = set()

    for i in I:
        for k in range(i + 1):  # k <= i
            if i == k:
                Delta_red.add((i, k))
            else:
                job_i = sorted_jobs[i]
                job_k = sorted_jobs[k]

                c_i = job_i[2]
                c_k = job_k[2]

                if intervals_overlap(job_i, job_k) and c_i + c_k > C:
                    # Coppia impossibile: non creo x[i,k]
                    continue

                Delta_red.add((i, k))

    # ------------------------------------------------------------
    # 4. Costruzione di Delta_nd_red
    # ------------------------------------------------------------
    # Riduzione per dominanza temporale.
    #
    # Se due job hanno lo stesso start time, per uno stesso k
    # può bastare tenere il vincolo di capacità del job con indice più alto.
    #
    # Per semplicità implementiamo la versione sicura:
    # per ogni coppia (i,k), se esiste j > i con stesso start time
    # e (j,k) in Delta_red, allora (i,k) è dominato.
    #
    # Questa riduzione si usa solo sui vincoli di capacità.

    Delta_nd_red = set()

    for (i, k) in Delta_red:
        if i == k:
            Delta_nd_red.add((i, k))
            continue

        s_i = sorted_jobs[i][0]

        dominated = False

        for j in range(i + 1, n):
            s_j = sorted_jobs[j][0]

            if s_j != s_i:
                # Poiché i job sono ordinati per start time,
                # appena cambia start time possiamo fermarci.
                if s_j > s_i:
                    break

            if s_j == s_i and (j, k) in Delta_red:
                dominated = True
                break

        if not dominated:
            Delta_nd_red.add((i, k))

    # ------------------------------------------------------------
    # 5. Creazione del modello Gurobi
    # ------------------------------------------------------------

    model = gp.Model("TBPP_FU_Model3")

    if not verbose:
        model.setParam("OutputFlag", 0)

    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)

    # ------------------------------------------------------------
    # 6. Variabili
    # ------------------------------------------------------------
    # x[i,k] = 1 se il job i va sul server inizializzato dal job k
    # w[i]   = 1 se il job i causa un fire-up

    x = model.addVars(
        list(Delta_red),
        vtype=GRB.BINARY,
        name="x"
    )

    w = model.addVars(
        list(I),
        vtype=GRB.BINARY,
        name="w"
    )

    # ------------------------------------------------------------
    # 7. Funzione obiettivo
    # ------------------------------------------------------------
    # Minimizziamo:
    #     gamma * numero fire-up + numero server usati
    #
    # Nel Modello 3:
    #     numero fire-up = sum_i w[i]
    #     numero server usati = sum_k x[k,k]

    model.setObjective(
        gamma * gp.quicksum(w[i] for i in I)
        + gp.quicksum(x[k, k] for k in I),
        GRB.MINIMIZE
    )

    # ------------------------------------------------------------
    # 8. Vincolo di assegnamento
    # ------------------------------------------------------------
    # Ogni job i deve essere assegnato esattamente a un server.
    #
    # sum_k x[i,k] = 1

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
    # 9. Vincoli di capacità
    # ------------------------------------------------------------
    # Per ogni coppia utile (i,k):
    #
    # sum_{j in delta_i} c_j x[j,k] + c_i x[i,k] <= C x[k,k]
    #
    # Escludiamo i == k perché darebbe un vincolo banale:
    #     c_i x[i,i] <= C x[i,i]

    for (i, k) in Delta_nd_red:
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
    # 10. Vincoli di collegamento
    # ------------------------------------------------------------
    # Se i va sul server inizializzato da k,
    # allora k deve davvero inizializzare quel server.
    #
    # x[i,k] <= x[k,k]

    for (i, k) in Delta_red:
        if i != k:
            model.addConstr(
                x[i, k] <= x[k, k],
                name=f"link_{i}_{k}"
            )

    # ------------------------------------------------------------
    # 11. Vincoli sui fire-up
    # ------------------------------------------------------------
    # Se il job i va sul server k e prima di i non c'era niente
    # su quel server, allora w[i] deve diventare 1.
    #
    # sum_{j in delta_plus_i} x[j,k] - x[i,k] + w[i] >= 0

    for (i, k) in Delta_red:
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
    # 12. Risoluzione
    # ------------------------------------------------------------

    model.optimize()

    # ------------------------------------------------------------
    # 13. Estrazione soluzione
    # ------------------------------------------------------------

    result = {
        "status": model.Status,
        "objective": None,
        "fireups": None,
        "servers_used": None,
        "assignment": None,
        "sorted_jobs": sorted_jobs,
        "original_ids": original_ids,
        "runtime": model.Runtime, # necessario per lo studio di scalabilità
        "model": model,
        "num_vars": model.NumVars,
        "num_constrs": model.NumConstrs,
    }

    if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        if model.SolCount > 0:
            result["objective"] = model.ObjVal

            fireup_jobs = []
            for i in I:
                if w[i].X > 0.5:
                    fireup_jobs.append(i)

            initializer_jobs = []
            for k in I:
                if (k, k) in Delta_red and x[k, k].X > 0.5:
                    initializer_jobs.append(k)

            result["fireups"] = len(fireup_jobs)
            result["servers_used"] = len(initializer_jobs)

            # assignment:
            # chiave = job inizializzatore k
            # valore = lista di job assegnati al server inizializzato da k
            assignment_sorted = {k: [] for k in initializer_jobs}

            for i in I:
                for k in range(i + 1):
                    if (i, k) in Delta_red and x[i, k].X > 0.5:
                        assignment_sorted.setdefault(k, []).append(i)
                        break

            # Convertiamo dagli indici ordinati agli indici originali
            assignment_original = {}

            for k, assigned_jobs in assignment_sorted.items():
                original_initializer = original_ids[k]
                original_assigned_jobs = [original_ids[i] for i in assigned_jobs]

                assignment_original[original_initializer] = original_assigned_jobs

            result["assignment"] = assignment_original

    return result


if __name__ == "__main__":
    # Esempio piccolo di test
    jobs = [
        (0, 4, 40),
        (1, 5, 30),
        (4, 7, 50),
        (6, 9, 40),
    ]

    C = 100

    result = solve_model3(
        jobs=jobs,
        C=C,
        gamma=1.0,
        time_limit=1800,
        verbose=True
    )

    print("\n--- RISULTATO MODELLO 3 ---")
    print("Status:", result["status"])
    print("Objective:", result["objective"])
    print("Server usati:", result["servers_used"])
    print("Fire-up:", result["fireups"])
    print("Assignment:", result["assignment"])