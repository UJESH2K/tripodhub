"""Notebook-friendly access to the tripodhub ML lab programs."""

from __future__ import annotations

from . import lab01, lab02, lab03, lab04, lab05, lab06, lab07, lab08, lab09, lab10
from .cli import main
from .code_templates import get_code

LAB_MODULES = {
    1: lab01,
    2: lab02,
    3: lab03,
    4: lab04,
    5: lab05,
    6: lab06,
    7: lab07,
    8: lab08,
    9: lab09,
    10: lab10,
}

lab1 = lab01
lab2 = lab02
lab3 = lab03
lab4 = lab04
lab5 = lab05
lab6 = lab06
lab7 = lab07
lab8 = lab08
lab9 = lab09
lab10 = lab10


def list_labs() -> list[int]:
    return list(LAB_MODULES)


def run_lab(lab_number: int, **kwargs: object) -> object:
    if lab_number not in LAB_MODULES:
        raise ValueError("lab_number must be between 1 and 10.")
    return LAB_MODULES[lab_number].run(**kwargs)


__all__ = [
    "lab1",
    "lab2",
    "lab3",
    "lab4",
    "lab5",
    "lab6",
    "lab7",
    "lab8",
    "lab9",
    "lab10",
    "lab01",
    "lab02",
    "lab03",
    "lab04",
    "lab05",
    "lab06",
    "lab07",
    "lab08",
    "lab09",
    "lab10",
    "get_code",
    "list_labs",
    "main",
    "run_lab",
]
