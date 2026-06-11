import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


Job = Tuple[int, int, int]  # (start, end, size)


@dataclass(frozen=True)
class TestbedBClassParams:
    a_min: int
    a_max: int
    b_min: int
    b_max: int


class TestbedBGenerator:
    """
    Generatore per il Testbed B secondo Dell'Amico et al. / Category B.

    Output:
        jobs = [(s_i, e_i, c_i), ...]

    Dove:
        s_i = primo time step in cui il job è attivo
        e_i = primo time step in cui il job NON è più attivo
        c_i = richiesta di capacità, uniforme in [10, 100]

    Nota:
        Le classi V e VII hanno gli stessi parametri perché così sono riportate
        nella tabella del paper.
    """

    C = 100

    CLASS_ORDER: List[str] = [
        "I", "II", "III", "IV", "V",
        "VI", "VII", "VIII", "IX", "X",
    ]

    CLASS_PARAMS: Dict[str, TestbedBClassParams] = {
        "I":    TestbedBClassParams(10, 10, 90, 95),
        "II":   TestbedBClassParams(15, 15, 90, 95),
        "III":  TestbedBClassParams(20, 20, 90, 95),
        "IV":   TestbedBClassParams(25, 25, 90, 95),
        "V":    TestbedBClassParams(30, 30, 90, 95),
        "VI":   TestbedBClassParams(30, 30, 70, 90),
        "VII":  TestbedBClassParams(30, 30, 90, 95),
        "VIII": TestbedBClassParams(25, 35, 90, 95),
        "IX":   TestbedBClassParams(25, 35, 70, 90),
        "X":    TestbedBClassParams(30, 40, 90, 95),
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_instance(self, T_size: int, class_name: str) -> List[Job]:
        """
        Genera una singola istanza del Testbed B.

        Parametri:
            T_size: numero di time step, ad esempio 5, 10, 15, 20, 30, ..., 150
            class_name: una classe tra "I", ..., "X"

        Ritorna:
            lista di job ordinati per start time: [(s, e, c), ...]
        """

        if class_name not in self.CLASS_PARAMS:
            raise ValueError(
                f"Classe non valida: {class_name}. "
                f"Classi valide: {self.CLASS_ORDER}"
            )

        #if T_size <= 0:
        if T_size < 2:
            raise ValueError("T_size deve essere maggiore di uno.")

        params = self.CLASS_PARAMS[class_name]

        active_sets: List[List[int]] = []
        weights: Dict[int, int] = {}

        next_job_id = 0
        previous_active: List[int] = []

        for t in range(T_size-1):
            num_active = self.rng.randint(params.a_min, params.a_max)

            if t == 0:
                inherited: List[int] = []
            else:
                beta = self.rng.randint(params.b_min, params.b_max)

                # Numero di job ereditati dal time step precedente.
                # Uso int(), cioè troncamento verso il basso, per evitare il
                # comportamento particolare di round() in Python sui casi .5.
                num_inherited = int(beta * len(previous_active) / 100)

                num_inherited = max(0, num_inherited)
                num_inherited = min(
                    num_inherited,
                    num_active,
                    len(previous_active),
                )

                inherited = self.rng.sample(previous_active, num_inherited)

            num_new = num_active - len(inherited)

            new_jobs = []
            for _ in range(num_new):
                job_id = next_job_id
                next_job_id += 1

                weights[job_id] = self.rng.randint(10, 100)
                new_jobs.append(job_id)

            current_active = inherited + new_jobs
            active_sets.append(current_active)
            previous_active = current_active

        return self._active_sets_to_jobs(active_sets, weights)

    def generate_testbed(
        self,
        T_values: List[int],
        instances_per_class: int = 10,
        base_seed: int = 0,
    ) -> Dict[Tuple[int, str, int], List[Job]]:
        """
        Genera un intero testbed.

        Chiave del dizionario:
            (T_size, class_name, instance_id)

        Esempio:
            testbed[(20, "III", 4)] -> quarta istanza della classe III con |T|=20

        Nota sui seed:
            Non viene usato hash(class_name), perché l'hash di Python può
            cambiare tra esecuzioni diverse. Il seed qui è deterministico.
        """

        if not T_values:
            raise ValueError("T_values non può essere vuoto.")

        testbed: Dict[Tuple[int, str, int], List[Job]] = {}

        for T_size in T_values:
            if T_size <= 0:
                raise ValueError("Tutti i valori di T_values devono essere positivi.")

            for class_name in self.CLASS_ORDER:
                class_index = self.CLASS_ORDER.index(class_name)

                for instance_id in range(instances_per_class):
                    seed = self.stable_seed(
                        base_seed=base_seed,
                        T_size=T_size,
                        class_name=class_name,
                        instance_id=instance_id,
                    )

                    generator = TestbedBGenerator(seed=seed)
                    jobs = generator.generate_instance(
                        T_size=T_size,
                        class_name=class_name,
                    )

                    testbed[(T_size, class_name, instance_id)] = jobs

        return testbed

    @classmethod
    def stable_seed(
        cls,
        base_seed: int,
        T_size: int,
        class_name: str,
        instance_id: int,
    ) -> int:
        """
        Seed stabile e riproducibile per una specifica istanza.
        """

        if class_name not in cls.CLASS_PARAMS:
            raise ValueError(f"Classe non valida: {class_name}")

        class_index = cls.CLASS_ORDER.index(class_name)
        return base_seed + 100000 * T_size + 1000 * class_index + instance_id

    @staticmethod
    def _active_sets_to_jobs(
        active_sets: List[List[int]],
        weights: Dict[int, int],
    ) -> List[Job]:
        starts: Dict[int, int] = {}
        ends: Dict[int, int] = {}

        for t, active in enumerate(active_sets):
            for job_id in active:
                if job_id not in starts:
                    starts[job_id] = t
                ends[job_id] = t + 1

        jobs = [
            (starts[job_id], ends[job_id], weights[job_id])
            for job_id in starts
        ]

        jobs.sort(key=lambda job: (job[0], job[1], job[2]))
        return jobs
