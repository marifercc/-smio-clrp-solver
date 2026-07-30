#!/usr/bin/env python3
"""Busqueda MAS AGRESIVA que depot_neighbors.py: en vez de mover un solo
deposito (abrir o cerrar uno), prueba TODOS los pares posibles de "cerrar
uno de los abiertos Y abrir uno de los cerrados" al mismo tiempo. Con
instancias grandes (30+ depositos) esto son cientos de combinaciones, asi
que primero se filtran con el proxy rapido de explore_depots2.py (sin
resolver rutas, solo una estimacion), y solo los mejores candidatos se
profundizan de verdad con PyVRP.

Uso:
    python solver/depot_2swap.py Instancias_oficiales/clrp-medium-07.txt out/pyvrp/medium-07_deep3.sol.txt --top 6 --deep-time 150 --seed 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clrp_solver import parse_instance, ensure_distance_cache, write_solution, compute_total_cost
from pyvrp_joint import solve_joint, parse_sol_routes
from explore_depots2 import quick_cost

_REPO_SRC = Path(__file__).resolve().parent.parent / "smio_clrp_verify_src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
from smio_clrp_verify.verify import verify_solution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=Path)
    ap.add_argument("input_sol", type=Path)
    ap.add_argument("--top", type=int, default=6, help="cuantos pares profundizar con PyVRP")
    ap.add_argument("--deep-time", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sol-dir", type=Path, default=Path("out/pyvrp"))
    args = ap.parse_args()

    inst = parse_instance(args.instance)
    ensure_distance_cache(inst)
    dist = np.asarray(inst._dist_cache, dtype=np.float64)
    m, n = inst.m, inst.n
    total_demand = sum(inst.demands)

    original_routes = parse_sol_routes(args.input_sol, m)
    base_combo = sorted(original_routes.keys())
    base_cost = compute_total_cost(inst, original_routes)
    closed = sorted(set(range(m)) - set(base_combo))

    print(f"Base: depositos {[d + 1 for d in base_combo]}, costo real={base_cost:.2f}")
    print(f"Evaluando {len(base_combo)} x {len(closed)} = {len(base_combo) * len(closed)} pares "
          f"(cerrar uno abierto + abrir uno cerrado) con el proxy rapido...")

    t0 = time.time()
    results = []
    for out_d in base_combo:
        for in_d in closed:
            combo = tuple(sorted((set(base_combo) - {out_d}) | {in_d}))
            total_cap = sum(min(inst.depot_capacities[d], inst.depot_max_vehicles[d] * inst.vehicle_capacity)
                             for d in combo)
            if total_cap < total_demand:
                continue
            cost = quick_cost(inst, dist, combo)
            results.append((cost, combo, out_d, in_d))

    results.sort(key=lambda t: t[0])
    print(f"{len(results)} pares factibles evaluados en {time.time() - t0:.1f}s (proxy rapido).")
    print("Top candidatos (proxy, SIN optimizar rutas todavia):")
    for cost, combo, out_d, in_d in results[: args.top]:
        print(f"  cerrar {out_d + 1}, abrir {in_d + 1} -> proxy={cost:.2f}")

    print("\nProfundizando los mejores con PyVRP (esto SI tarda)...")
    best = (base_cost, base_combo, original_routes)
    customers = list(range(n))
    args.sol_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = args.sol_dir / f"_tmp_2swap_{args.instance.stem}.sol.txt"

    for i, (proxy_cost, combo, out_d, in_d) in enumerate(results[: args.top]):
        t1 = time.time()
        routes = solve_joint(inst, list(combo), customers, args.deep_time, args.seed + i)
        elapsed = time.time() - t1
        if routes is None:
            print(f"  cerrar {out_d + 1}, abrir {in_d + 1}: PyVRP infactible (tiempo {elapsed:.0f}s)")
            continue
        real_cost = compute_total_cost(inst, routes)
        if real_cost >= best[0] - 1e-6:
            print(f"  cerrar {out_d + 1}, abrir {in_d + 1}: costo={real_cost:.2f} (tiempo {elapsed:.0f}s)")
            continue
        write_solution(inst, routes, tmp_path)
        check = verify_solution(inst, tmp_path)
        if not check.feasible:
            print(f"  cerrar {out_d + 1}, abrir {in_d + 1}: costo={real_cost:.2f} pero el verificador "
                  f"OFICIAL lo marca INFACTIBLE -- se descarta (tiempo {elapsed:.0f}s)")
            continue
        print(f"  cerrar {out_d + 1}, abrir {in_d + 1}: costo={real_cost:.2f} (tiempo {elapsed:.0f}s)  "
              f"<-- MEJOR HASTA AHORA")
        best = (real_cost, combo, routes)

    tmp_path.unlink(missing_ok=True)

    cost, combo, routes = best
    out_path = args.sol_dir / f"{args.instance.stem}_2swap.sol.txt"
    write_solution(inst, routes, out_path)
    result = verify_solution(inst, out_path)
    print(f"\nMEJOR FINAL: depositos {[d + 1 for d in combo]}, costo={cost:.2f}")
    print(f"Guardado en {out_path} -- feasible={result.feasible} recomputed_cost={result.recomputed_cost:.2f}")


if __name__ == "__main__":
    main()
