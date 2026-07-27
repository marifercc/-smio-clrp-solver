#!/usr/bin/env python3
"""Busca vecinos del conjunto de depositos abiertos ACTUAL (quitar uno, o
agregar uno de los cerrados) y profundiza cada vecino con PyVRP, para ver si
mover UN deposito baja el costo real por debajo del mejor que ya tenemos.

Complementa a explore_depots2.py: aquel explora TODO el espacio con un proxy
rapido (que resulto sesgado hacia abrir de mas depositos); este ataca
puntualmente los vecinos mas cercanos a la combinacion que ya sabemos que
funciona bien.

Uso:
    python solver/depot_neighbors.py Instancias_oficiales/clrp-small-03.txt out/pyvrp/small-03_joint.sol.txt --time-limit 90 --seed 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clrp_solver import parse_instance, ensure_distance_cache, write_solution, compute_total_cost
from pyvrp_joint import solve_joint, parse_sol_routes

_REPO_SRC = Path(__file__).resolve().parent.parent / "smio_clrp_verify_src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
from smio_clrp_verify.verify import verify_solution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=Path)
    ap.add_argument("input_sol", type=Path)
    ap.add_argument("--time-limit", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sol-dir", type=Path, default=Path("out/pyvrp"))
    args = ap.parse_args()

    inst = parse_instance(args.instance)
    ensure_distance_cache(inst)
    m, n = inst.m, inst.n

    original_routes = parse_sol_routes(args.input_sol, m)
    base_combo = sorted(original_routes.keys())
    base_cost = compute_total_cost(inst, original_routes)
    closed = sorted(set(range(m)) - set(base_combo))

    print(f"Base: depositos {[d + 1 for d in base_combo]}, costo={base_cost:.2f}")

    neighbors = []
    for d in base_combo:
        neighbors.append(("quitar", tuple(sorted(set(base_combo) - {d})), d))
    for d in closed:
        neighbors.append(("agregar", tuple(sorted(set(base_combo) | {d})), d))

    best = (base_cost, base_combo, original_routes)
    customers = list(range(n))
    total_demand = sum(inst.demands)
    args.sol_dir.mkdir(parents=True, exist_ok=True)

    for i, (accion, combo, moved) in enumerate(neighbors):
        total_cap = sum(min(inst.depot_capacities[d], inst.depot_max_vehicles[d] * inst.vehicle_capacity)
                         for d in combo)
        if total_cap < total_demand:
            print(f"  {accion} deposito {moved + 1} -> {[d + 1 for d in combo]}: capacidad insuficiente, se salta")
            continue
        t0 = time.time()
        routes = solve_joint(inst, list(combo), customers, args.time_limit, args.seed + i)
        if routes is None:
            print(f"  {accion} deposito {moved + 1} -> {[d + 1 for d in combo]}: PyVRP infactible "
                  f"(tiempo {time.time() - t0:.0f}s)")
            continue
        cost = compute_total_cost(inst, routes)
        marker = "  <-- MEJOR HASTA AHORA" if cost < best[0] - 1e-6 else ""
        print(f"  {accion} deposito {moved + 1} -> {[d + 1 for d in combo]}: costo={cost:.2f} "
              f"(tiempo {time.time() - t0:.0f}s){marker}")
        if cost < best[0] - 1e-6:
            best = (cost, combo, routes)

    cost, combo, routes = best
    out_path = args.sol_dir / f"{args.instance.stem}_neighbors.sol.txt"
    write_solution(inst, routes, out_path)
    result = verify_solution(inst, out_path)
    print(f"\nMEJOR FINAL: depositos {[d + 1 for d in combo]}, costo={cost:.2f}")
    print(f"Guardado en {out_path} -- feasible={result.feasible} recomputed_cost={result.recomputed_cost:.2f}")


if __name__ == "__main__":
    main()
