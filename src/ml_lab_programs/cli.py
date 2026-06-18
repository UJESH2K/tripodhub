"""Command-line runner for ML lab programs."""

from __future__ import annotations

import argparse
from typing import Callable

from . import lab01, lab02, lab03, lab04, lab05, lab06, lab07, lab08, lab09, lab10


LAB_RUNNERS: dict[str, Callable[..., object]] = {
    "lab01": lab01.run,
    "lab02": lab02.run,
    "lab03": lab03.run,
    "lab04": lab04.run,
    "lab05": lab05.run,
    "lab06": lab06.run,
    "lab07": lab07.run,
    "lab08": lab08.run,
    "lab09": lab09.run,
    "lab10": lab10.run,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ML lab programs.")
    parser.add_argument("lab", nargs="?", choices=LAB_RUNNERS.keys(), help="Lab module to run")
    parser.add_argument("--list", action="store_true", help="List available lab modules")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting where supported")
    parser.add_argument("--csv-path", help="CSV path for lab04 Find-S input")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list or not args.lab:
        print("\n".join(LAB_RUNNERS))
        return

    runner = LAB_RUNNERS[args.lab]

    if args.lab == "lab04":
        result = runner(csv_path=args.csv_path)
    elif args.lab in {"lab01", "lab02", "lab03", "lab05", "lab06", "lab07", "lab10"}:
        result = runner(show_plot=not args.no_plots) if args.lab in {"lab03", "lab05", "lab06", "lab07", "lab10"} else runner(show_plots=not args.no_plots)
    else:
        result = runner()

    print(result)


if __name__ == "__main__":
    main()
