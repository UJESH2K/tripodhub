"""Command-line runner for tripodhub labs."""

from __future__ import annotations

import argparse
from typing import Callable

from . import lab01, lab02, lab03, lab04, lab05, lab06, lab07, lab08, lab09, lab10
from .code_templates import get_code


LAB_RUNNERS: dict[int, Callable[..., object]] = {
    1: lab01.run,
    2: lab02.run,
    3: lab03.run,
    4: lab04.run,
    5: lab05.run,
    6: lab06.run,
    7: lab07.run,
    8: lab08.run,
    9: lab09.run,
    10: lab10.run,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print or run tripodhub ML lab programs.")
    parser.add_argument("lab", nargs="?", type=int, choices=range(1, 11), help="Lab number to run")
    parser.add_argument("--list", action="store_true", help="List available lab numbers")
    parser.add_argument("--run", action="store_true", help="Execute the lab instead of printing notebook code")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting where supported")
    parser.add_argument("--csv-path", help="CSV path for lab 4 Find-S input")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list or args.lab is None:
        print("\n".join(str(number) for number in LAB_RUNNERS))
        return

    if not args.run:
        print(get_code(args.lab))
        return

    runner = LAB_RUNNERS[args.lab]

    if args.lab == 4:
        result = runner(csv_path=args.csv_path)
    elif args.lab in {1, 2}:
        result = runner(show_plots=not args.no_plots)
    elif args.lab in {3, 5, 6, 7, 10}:
        result = runner(show_plot=not args.no_plots)
    else:
        result = runner()

    print(result)


if __name__ == "__main__":
    main()
