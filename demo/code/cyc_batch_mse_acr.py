import argparse
import subprocess
import sys
from pathlib import Path


def run_once(script: Path, dt: float, cwd: Path) -> None:
	"""Run a target script with a specific dt."""
	cmd = [sys.executable, str(script), "--dt", f"{dt:.3f}"]
	print(f"[RUN] dt={dt:.3f} -> {script.name}")
	subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
	parser = argparse.ArgumentParser(description="Batch runner for ACR and MSE jobs.")
	parser.add_argument(
		"--dts",
		nargs="+",
		type=float,
		default=[i / 10 for i in range(1, 11)],
		help="List of dt values to sweep (default: 0.1 ... 1.0).",
	)
	args = parser.parse_args()

	code_dir = Path(__file__).resolve().parent
	acr_script = code_dir / "cyc_attractor_consistency.py"
	mse_script = code_dir / "cyc_compute_mse_nrmse_r2.py"

	for dt in args.dts:
		run_once(acr_script, dt, cwd=code_dir)
		run_once(mse_script, dt, cwd=code_dir)


if __name__ == "__main__":
	main()
