import random

def generate_testbed_B(T_size, a_min, a_max, b_min, b_max, seed=None):
    """
    Genera istanze basate sul Testbed B (Dell'Amico et al., 2020).
    
    T_size: numero totale di istanti temporali |T|
    a_min, a_max: range per il numero di job attivi in ogni istante
    b_min, b_max: range percentuale [0-100] dei job 'ereditati' (sopravvissuti)
    """
    if seed is not None:
        random.seed(seed)
        
    C = 100
    all_jobs_completed = []  # Qui salveremo i job definitivi (start, end, capacity)
    active_jobs = []         # Lista temporanea dei job attualmente in esecuzione
    
    # Ciclo attraverso ogni istante di tempo da 1 fino a |T|
    for t in range(1, T_size + 1):
        
        # 1. Quanti job attivi servono in questo preciso istante?
        n_active_target = random.randint(a_min, a_max)
        
        if t == 1:
            # Primo istante: nessun job ereditato, si creano tutti da zero
            for _ in range(n_active_target):
                c = random.randint(10, 100) # Capacità uniforme tra 10 e 100
                active_jobs.append({"start": t, "capacity": c})
                
        else:
            # 2. Istanti successivi: calcoliamo la percentuale di eredità
            p_inherit = random.randint(b_min, b_max) / 100.0
            
            # Quanti job sopravvivono? (Arrotondato all'intero)
            k_inherit = int(round(p_inherit * len(active_jobs)))
            
            # Sicurezze: non possiamo ereditare più job di quanti ne avevamo, 
            # né più di quanti ce ne servono ora
            k_inherit = min(k_inherit, len(active_jobs), n_active_target)
            
            # 3. Selezioniamo casualmente i fortunati che sopravvivono
            inherited = random.sample(active_jobs, k_inherit)
            
            # 4. Chi NON è stato selezionato termina la sua esecuzione al tempo t
            for job in active_jobs:
                if job not in inherited:
                    # Salviamo il job completato: (start, end, capacity)
                    all_jobs_completed.append((job["start"], t, job["capacity"]))
                    
            # Aggiorniamo la lista degli attivi solo con quelli sopravvissuti
            active_jobs = inherited
            
            # 5. Creiamo nuovi job per raggiungere il target richiesto (n_active_target)
            n_new = n_active_target - k_inherit
            for _ in range(n_new):
                c = random.randint(10, 100)
                active_jobs.append({"start": t, "capacity": c})
                
    # 6. Fine orizzonte: tutti i job ancora attivi terminano all'istante successivo
    for job in active_jobs:
        all_jobs_completed.append((job["start"], T_size + 1, job["capacity"]))
        
    # Per pulizia, ordiniamo i job in base al tempo di inizio
    all_jobs_completed.sort(key=lambda x: (x[0], x[1]))
    
    return all_jobs_completed

# --- TEST VELOCE ---
if __name__ == "__main__":
    # Proviamo con i parametri di una ipotetica "Classe" dell'articolo
    jobs = generate_testbed_B(
        T_size=15, 
        a_min=5, a_max=10,   # Da 5 a 10 job in esecuzione in ogni istante
        b_min=70, b_max=90,  # Alta percentuale di eredità (i job durano a lungo)
        seed=42
    )
    
    print(f"Generati {len(jobs)} job totali per il Testbed B.")
    for i, job in enumerate(jobs):
        print(f"Job {i}: start={job[0]:2d}, end={job[1]:2d}, cap={job[2]:3d}")