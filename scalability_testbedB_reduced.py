"""
Scalability test - Testbed B - modelli ridotti/ottimizzati M1, M2, M3.

Questo file NON ridefinisce il generatore del Testbed B.
Importa invece la classe TestbedBGenerator dal file separato:

    testbed_b_generator.py

Output generati:
    results_testbedB_reduced/testbedB_results.csv
    results_testbedB_reduced/testbedB_scalability_runtime.png
    results_testbedB_reduced/testbedB_scalability_optimal_only.png
"""

import os
import csv
import time
import statistics
from typing import Dict, List, Tuple, Any, Optional, Callable

import matplotlib.pyplot as plt

from testbed_b_generator import TestbedBGenerator

from modello1.model1_optimized_tbpp_fu import solve_model1_optimized
from modello2.model2_optimized_tbpp_fu import solve_model2_optimized
from modello3.model3_optimized_tbpp_fu import solve_model3_optimized


Job = Tuple[int, int, int]
Solver = Callable[..., Dict[str, Any]]


class TestbedBScalabilityTest:
    def __init__(
        self,
        T_values: List[int],
        classes: Optional[List[str]] = None,
        instances_per_class: int = 10,
        C: int = 100,
        gamma: float = 1.0,
        time_limit: int = 1800,
        output_dir: str = "results_testbedB_reduced",
        base_seed: int = 42,
    ):
        self.T_values = T_values
        self.classes = classes if classes is not None else list(TestbedBGenerator.CLASS_ORDER)
        self.instances_per_class = instances_per_class
        self.C = C
        self.gamma = gamma
        self.time_limit = time_limit
        self.output_dir = output_dir
        self.base_seed = base_seed

        os.makedirs(self.output_dir, exist_ok=True)

        self.csv_path = os.path.join(self.output_dir, "testbedB_results.csv")
        self.plot_all_path = os.path.join(self.output_dir, "testbedB_scalability_runtime.png")
        self.plot_optimal_path = os.path.join(self.output_dir, "testbedB_scalability_optimal_only.png")

        self.models: Dict[str, Solver] = {
            "M1 reduced": solve_model1_optimized,
            "M2 reduced": solve_model2_optimized,
            "M3 reduced": solve_model3_optimized,
        }

        self._validate_input()

    def _validate_input(self) -> None:
        if not self.T_values:
            raise ValueError("T_values non può essere vuoto.")

        if any(T_size <= 0 for T_size in self.T_values):
            raise ValueError("Tutti i valori di T_values devono essere positivi.")

        if self.instances_per_class <= 0:
            raise ValueError("instances_per_class deve essere positivo.")

        valid_classes = set(TestbedBGenerator.CLASS_PARAMS.keys())
        for class_name in self.classes:
            if class_name not in valid_classes:
                raise ValueError(
                    f"Classe non valida: {class_name}. "
                    f"Classi valide: {TestbedBGenerator.CLASS_ORDER}"
                )

    def _stable_seed(self, T_size: int, class_name: str, instance_id: int) -> int:
        return TestbedBGenerator.stable_seed(
            base_seed=self.base_seed,
            T_size=T_size,
            class_name=class_name,
            instance_id=instance_id,
        )

    def run(self) -> None:
        rows: List[Dict[str, Any]] = []

        print("=" * 80)
        print("SCALABILITY TEST - TESTBED B - MODELLI RIDOTTI")
        print("=" * 80)
        print(f"T_values: {self.T_values}")
        print(f"Classes: {self.classes}")
        print(f"Instances per class: {self.instances_per_class}")
        print(f"C: {self.C}")
        print(f"gamma: {self.gamma}")
        print(f"Time limit per modello/istanza: {self.time_limit} s")
        print(f"Output dir: {self.output_dir}")
        print("=" * 80)

        for T_size in self.T_values:
            for class_name in self.classes:
                for instance_id in range(self.instances_per_class):
                    seed = self._stable_seed(T_size, class_name, instance_id)
                    generator = TestbedBGenerator(seed=seed)
                    jobs = generator.generate_instance(T_size=T_size, class_name=class_name)
                    n_jobs = len(jobs)

                    print(
                        f"\n--- |T|={T_size}, class={class_name}, "
                        f"instance={instance_id}, seed={seed}, n_jobs={n_jobs} ---"
                    )

                    for model_name, solver in self.models.items():
                        result = self._solve_one_model(solver, jobs)

                        row = {
                            "T_size": T_size,
                            "class": class_name,
                            "instance_id": instance_id,
                            "seed": seed,
                            "n_jobs": n_jobs,
                            "model": model_name,
                            "runtime": result["runtime"],
                            "status": result["status"],
                            "obj": result["obj"],
                            "optimal": result["optimal"],
                            "error": result["error"],
                        }
                        rows.append(row)

                        print(
                            f"{model_name:10s} | "
                            f"runtime={result['runtime']:9.3f} s | "
                            f"optimal={str(result['optimal']):5s} | "
                            f"status={result['status']} | "
                            f"obj={result['obj']}"
                        )

                    # Salvataggio progressivo: utile se il test viene interrotto.
                    self._save_csv(rows)

        self._save_csv(rows)
        self._plot_all_runs(rows)
        self._plot_optimal_only(rows)

        print("\nRisultati salvati in:")
        print(f"CSV:        {self.csv_path}")
        print(f"Grafico 1:  {self.plot_all_path}")
        print(f"Grafico 2:  {self.plot_optimal_path}")

    def _solve_one_model(self, solver: Solver, jobs: List[Job]) -> Dict[str, Any]:
        start = time.perf_counter()

        try:
            result = solver(
                jobs=jobs,
                C=self.C,
                gamma=self.gamma,
                time_limit=self.time_limit,
                verbose=False
            )

            elapsed = time.perf_counter() - start

            if result is None:
                return {
                    "runtime": elapsed,
                    "status": "NO_RESULT",
                    "obj": None,
                    "optimal": False,
                    "error": "Il solver ha restituito None.",
                }

            status = result.get("status", None)
            obj = result.get("objective", result.get("obj", None))
            optimal = self._is_optimal_status(status)

            return {
                "runtime": elapsed,
                "status": status,
                "obj": obj,
                "optimal": optimal,
                "error": "",
            }

        except Exception as e:
            elapsed = time.perf_counter() - start
            return {
                "runtime": elapsed,
                "status": "ERROR",
                "obj": None,
                "optimal": False,
                "error": repr(e),
            }

    @staticmethod
    def _is_optimal_status(status: Any) -> bool:
        if status == 2:
            return True
        if isinstance(status, str) and status.upper() == "OPTIMAL":
            return True
        return False

    def _save_csv(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        fieldnames = [
            "T_size",
            "class",
            "instance_id",
            "seed",
            "n_jobs",
            "model",
            "runtime",
            "status",
            "obj",
            "optimal",
            "error",
        ]

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _group_by_T_and_model(
        self,
        rows: List[Dict[str, Any]],
        optimal_only: bool = False,
    ) -> Dict[Tuple[int, str], Dict[str, List[float]]]:
        grouped: Dict[Tuple[int, str], Dict[str, List[float]]] = {}

        for row in rows:
            if optimal_only and not row["optimal"]:
                continue

            key = (int(row["T_size"]), str(row["model"]))
            grouped.setdefault(key, {"runtimes": [], "n_jobs": []})
            grouped[key]["runtimes"].append(float(row["runtime"]))
            grouped[key]["n_jobs"].append(int(row["n_jobs"]))

        return grouped

    def _plot_all_runs(self, rows: List[Dict[str, Any]]) -> None:
        """
        Grafico principale: usa tutte le run.
        Se una run arriva al time limit, il suo tempo misurato entra nella media.
        Questo è utile per vedere il costo computazionale reale del test.
        """
        self._plot(
            rows=rows,
            output_path=self.plot_all_path,
            title="Scalability test - Testbed B - modelli ridotti",
            optimal_only=False,
        )

    def _plot_optimal_only(self, rows: List[Dict[str, Any]]) -> None:
        """
        Grafico secondario: usa solo le run risolte a ottimalità.
        Utile se vuoi confrontarti con tabelle in cui i time limit vengono esclusi.
        """
        self._plot(
            rows=rows,
            output_path=self.plot_optimal_path,
            title="Scalability test - Testbed B - solo istanze ottime",
            optimal_only=True,
        )

    def _plot(
        self,
        rows: List[Dict[str, Any]],
        output_path: str,
        title: str,
        optimal_only: bool,
    ) -> None:
        grouped = self._group_by_T_and_model(rows, optimal_only=optimal_only)
        models = list(self.models.keys())

        plt.figure(figsize=(10, 6))

        for model in models:
            xs = []
            ys = []

            for T_size in sorted(self.T_values):
                key = (T_size, model)
                if key not in grouped or not grouped[key]["runtimes"]:
                    continue

                mean_n_jobs = statistics.mean(grouped[key]["n_jobs"])
                mean_runtime = statistics.mean(grouped[key]["runtimes"])

                xs.append(mean_n_jobs)
                ys.append(mean_runtime)

            if xs and ys:
                plt.plot(xs, ys, marker="o", label=model)

        plt.xlabel("Numero medio di job generati")
        plt.ylabel("Runtime medio [s]")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    # Setup leggero per prova rapida.
    # Per avvicinarti al paper usa, ad esempio:
    # T_values=[10, 15, 20, 30, 40, 50, 60]
    # instances_per_class=10
    test = TestbedBScalabilityTest(
        T_values=[10, 15],
        classes=["I", "III", "V"],
        instances_per_class=3,
        time_limit=1800,
        output_dir="results_testbedB_reduced",
        base_seed=42,
    )

    test.run()
