#!/usr/bin/env python3
"""Busqueda de subconjuntos de depositos para CUALQUIER instancia, con un
proxy de costo rapido y siempre-seguro (nunca usa pack_routes ni ningun loop
de reparacion que se pueda tardar -- por eso no hace falta timeout/alarm, y
funciona igual en Windows) para filtrar candidatos, seguido de una
optimizacion PROFUNDA con PyVRP (pyvrp_joint.solve_joint, deja a PyVRP
reasignar clientes entre los depositos elegidos Y rutear) sobre los mejores
candidatos, para ver si algun subconjunto de depositos distinto al que ya
usamos da un costo real mas bajo.

Uso:
    python solver/explore_depots2.py Instancias_oficiales/clrp-small-03.txt --top 6 --deep-time 300 --seed 0
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clrp_solver import parse_instance, ensure_distance_cache, write_solution, compute_total_cost
from pyvrp_joint import solve_joint

_REPO_SRC = Path(__file__).resolve().parent.parent / "smio_clrp_verify_src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
from smio_clrp_verify.verify import verify_solution


def quick_cost(inst, dist: np.ndarray, combo: tuple[int, ...]) -> float:
    """Proxy rapido: asigna cada cliente a su deposito mas cercano DENTRO del
    combo (ignorando capacidad por completo), estima rutas = ceil(demanda/Q)
    por deposito (tope por depot_max_vehicles), y aproxima la distancia como
    ida-y-vuelta directa deposito-cliente. O(n * len(combo)); nunca hace
    loops de reparacion, siempre termina rapido."""
    m, n = inst.m, inst.n
    Q = inst.vehicle_capacity

    depot_idxs = np.array(combo)
    cust_idxs = np.arange(m, m + n)
    sub = dist[np.ix_(depot_idxs, cust_idxs)]  # len(combo) x n
    nearest_local = sub.argmin(axis=0)
    nearest_dist = sub[nearest_local, np.arange(n)]

    demands = np.asarray(inst.demands, dtype=np.float64)
    total = sum(inst.opening_costs[d] for d in combo)
    total += 2.0 * nearest_dist.sum()

    for local_idx, d in enumerate(combo):
        dem = demands[nearest_local == local_idx].sum()
        if dem <= 0:
            continue
        routes_needed = int(np.ceil(dem / Q))
        routes_needed = min(routes_needed, inst.depot_max_vehicles[d])
        total += routes_needed * inst.route_fixed_cost

    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=Path)
    ap.add_argument("--top", type=int, default=5, help="cuantos subconjuntos profundizar con PyVRP")
    ap.add_argument("--deep-time", type=float, default=300.0, help="segundos de PyVRP por candidato")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sol-dir", type=Path, default=Path("out/pyvrp"))
    args = ap.parse_args()

    inst = parse_instance(args.instance)
    ensure_distance_cache(inst)
    dist = np.asarray(inst._dist_cache, dtype=np.float64)
    m, n = inst.m, inst.n
    total_demand = sum(inst.demands)

    print(f"{inst.name}: {m} depositos, {n} clientes, explorando subconjuntos...")

    results = []
    t0 = time.time()
    for r in range(1, m + 1):
        for combo in itertools.combinations(range(m), r):
            total_cap = sum(min(inst.depot_capacities[d], inst.depot_max_vehicles[d] * inst.vehicle_capacity)
                             for d in combo)
            if total_cap < total_demand:
                continue
            cost = quick_cost(inst, dist, combo)
            results.append((cost, combo))

    results.sort(key=lambda t: t[0])
    print(f"{len(results)} subconjuntos factibles evaluados en {time.time()-t0:.1f}s (proxy rapido).")
    print("Top candidatos (costo proxy, SIN optimizar rutas todavia):")
    for cost, combo in results[: args.top]:
        print(f"  depositos {[d + 1 for d in combo]}: proxy={cost:.2f}")

    print("\nProfundizando los mejores con PyVRP (esto SI tarda, una corrida por candidato)...")
    best_overall = None
    args.sol_dir.mkdir(parents=True, exist_ok=True)
    for i, (proxy_cost, combo) in enumerate(results[: args.top]):
        customers = list(range(n))
        t1 = time.time()
        routes = solve_joint(inst, list(combo), customers, args.deep_time, args.seed + i)
        if routes is None:
            print(f"  depositos {[d + 1 for d in combo]}: PyVRP no encontro solucion factible "
                  f"(tiempo {time.time() - t1:.0f}s)")
            continue
        real_cost = compute_total_cost(inst, routes)
        print(f"  depositos {[d + 1 for d in combo]}: costo real={real_cost:.2f} "
              f"(proxy era {proxy_cost:.2f}, tiempo {time.time() - t1:.0f}s)")
        if best_overall is None or real_cost < best_overall[0]:
            best_overall = (real_cost, combo, routes)

    if best_overall is None:
        print("\nNingun candidato produjo una solucion factible.")
        return

    real_cost, combo, routes = best_overall
    out_path = args.sol_dir / f"{args.instance.stem}_depotsearch.sol.txt"
    write_solution(inst, routes, out_path)
    result = verify_solution(inst, out_path)
    print(f"\nMEJOR ENCONTRADO: depositos {[d + 1 for d in combo]}, costo={real_cost:.2f}")
    print(f"Guardado en {out_path} -- feasible={result.feasible} recomputed_cost={result.recomputed_cost:.2f}")


if __name__ == "__main__":
    main()
