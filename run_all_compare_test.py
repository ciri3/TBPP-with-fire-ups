import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

COMPARE_SCRIPTS = [
    "compare_base_vs_opt_testbedA.py",
    "compare_base_vs_opt_testbedB.py",
    "compare_opt_testbedA.py",
    "compare_opt_testbedB.py",
]


def run_script(script_name: str):
    script_path = PROJECT_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"File non trovato: {script_path}")

    print("\n" + "=" * 80)
    print(f"ESECUZIONE: {script_name}")
    print("=" * 80)

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print("\n" + "!" * 80)
        print(f"ERRORE durante l'esecuzione di: {script_name}")
        print(f"Exit code: {result.returncode}")
        print("Esecuzione interrotta.")
        print("!" * 80)
        sys.exit(result.returncode)

    print(f"\nCOMPLETATO: {script_name}")
    print(f"Tempo impiegato: {elapsed:.2f} secondi")


def main():
    print("=" * 80)
    print("RUN COMPLETO DEI TEST DI CONFRONTO / SCALABILITY")
    print("=" * 80)

    total_start = time.time()

    for script in COMPARE_SCRIPTS:
        run_script(script)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print("TUTTI I TEST SONO STATI COMPLETATI CORRETTAMENTE")
    print(f"Tempo totale: {total_elapsed:.2f} secondi")
    print("=" * 80)


if __name__ == "__main__":
    main()