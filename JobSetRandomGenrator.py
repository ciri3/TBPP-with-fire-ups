import random

def generate_jobs(
    n,
    C=100,
    s_factor=1.0,
    duration_type="short",
    size_type="low",
    seed=None
):
    if seed is not None:
        random.seed(seed)

    horizon_start_max = int(s_factor * n)

    if duration_type == "short":
        d_min, d_max = 10, 30
    elif duration_type == "long":
        d_min, d_max = 20, 60
    else:
        raise ValueError("duration_type deve essere 'short' oppure 'long'")

    if size_type == "low":
        c_min, c_max = 25, 50
    elif size_type == "high":
        c_min, c_max = 25, 75
    else:
        raise ValueError("size_type deve essere 'low' oppure 'high'")

    jobs = []

    for i in range(1, n + 1):
        s_i = random.randint(0, horizon_start_max)
        d_i = random.randint(d_min, d_max)
        e_i = s_i + d_i
        c_i = random.randint(c_min, c_max)

        jobs.append({
            "id": i,
            "s": s_i,
            "e": e_i,
            "c": c_i
        })

    jobs.sort(key=lambda job: (job["s"], job["e"], job["id"]))

    for new_id, job in enumerate(jobs, start=1):
        job["id"] = new_id

    return jobs