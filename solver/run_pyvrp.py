#!/usr/bin/env python3
"""Corre el motor PyVRP (fork LRP de Applied Routing, quickstart oficial del
reto) sobre una instancia oficial del reto SMIO. PyVRP es un solver de VRP en
C++ (Hybrid Genetic Search) mucho mas
fuerte que nuestro local search hecho a mano en Python -- en las instancias
mock del propio repo de referencia, le gana al baseline de Hexaly en 9 de 10
casos (~2% en promedio).

Estrategias disponibles (--strategy):
  direct        -- resuelve la instancia LRP completa directo con PyVRP.
  decomposition -- primero elige un subconjunto de depositos (el mas cercano
                   a cada cliente + los que hagan falta por capacidad, en
                   orden de costo fijo creciente), luego rutea solo esos con
                   PyVRP. Pensado para instancias grandes.
  multi_start   -- prueba varios subconjuntos de depositos (el greedy de
                   decomposition + 4 variantes al azar) con una corrida corta
                   cada uno, se queda con el mejor, y le da el tiempo restante
                   completo a ese subconjunto. Buena opcion por default.

Uso:
    python solver/run_pyvrp.py Instancias_oficiales/clrp-small-08.txt out/pyvrp/small-08.sol.txt --strategy multi_start --time-limit 300 --seed 0 --verify

Requisitos (instalar UNA sola vez -- ver INSTRUCCIONES_PYVRP.md):
    pip install numpy tqdm
    pip install "pyvrp @ https://github.com/valeriaarciga/LRP_2026/raw/main/pyvrp-0.13.3-cp312-cp312-win_amd64.whl"
    (el wheel es para Python 3.12 de 64 bits en Windows -- revisa tu version
    con `python --version` antes de instalar)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "pyvrp_lrp"))
from lrp import read, write  # noqa: E402
from lrp.strategies import STRATEGIES  # noqa: E402

_REPO_SRC = Path(__file__).resolve().parent.parent / "smio_clrp_verify_src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="multi_start")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    data = read(args.instance, scale=args.scale)
    result = STRATEGIES[args.strategy](data, args.time_limit, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, data, result, scale=args.scale)

    print(
        f"{args.instance.stem}: estrategia={args.strategy} "
        f"factible={result.is_feasible()} costo={result.cost() / args.scale:.2f} "
        f"iteraciones={result.num_iterations} tiempo={result.runtime:.1f}s"
    )

    if args.verify:
        from smio_clrp_verify.verify import parse_instance, verify_solution

        inst = parse_instance(args.instance)
        res = verify_solution(inst, args.output)
        print(
            f"verificador oficial: feasible={res.feasible} "
            f"costo_recalculado={res.recomputed_cost:.2f}"
        )
        for e in res.errors:
            print("  ERROR:", e)


if __name__ == "__main__":
    main()
