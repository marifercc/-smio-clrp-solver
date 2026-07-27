#!/usr/bin/env python3
"""Re-optimiza las RUTAS de una .sol.txt ya factible usando PyVRP (motor C++,
Hybrid Genetic Search) en vez de OR-Tools, dejando fija la decision de que
depositos estan abiertos y que cliente le toca a cada uno (eso ya lo decide
nuestro propio heuristico/facility-location). Es la misma idea que
reopt_routes.py pero con un motor de ruteo mas fuerte -- no necesita la
version modificada de PyVRP con costos de deposito, porque esa parte del
problema (abrir o no un deposito) ya esta resuelta antes de llegar aqui.

Por que es seguro: como el conjunto de clientes por deposito no cambia, la
factibilidad de capacidad de deposito y cobertura de clientes se hereda
automaticamente de la solucion de entrada -- solo puede cambiar el numero de
rutas y su orden, y solo si baja el costo. Si PyVRP no encuentra nada mejor
para un deposito, se conserva la solucion original de ese deposito tal cual.

Uso:
    python solver/pyvrp_route.py <instancia.txt> <entrada.sol.txt> <salida.sol.txt> [--time-per-depot 30] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyvrp
import pyvrp.stop

_REPO_SRC = Path(__file__).resolve().parent.parent / "smio_clrp_verify_src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clrp_solver import parse_instance, edge_cost, ensure_distance_cache, write_solution
from smio_clrp_verify.verify import verify_solution

SCALE = 10  # nuestras distancias/costos ya tienen 1 decimal -- escalamos a enteros para PyVRP


def parse_sol_routes(sol_path: Path, m: int) -> dict[int, list[list[int]]]:
    """{deposito_idx(0-based): [[cliente_idx(0-based), ...], ...]} desde un .sol.txt."""
    routes: dict[int, list[list[int]]] = {}
    current_depot = None
    for line in sol_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("DEPOT "):
            current_depot = int(line.split()[1]) - 1
            routes.setdefault(current_depot, [])
        elif line.startswith("ROUTE :"):
            ids = [int(x) for x in line.split(":")[1].split()]
            routes[current_depot].append([cid - 1 - m for cid in ids])
    return routes


def depot_route_cost(inst, depot: int, routes: list[list[int]]) -> float:
    total = 0.0
    for r in routes:
        nodes = [depot] + [inst.m + c for c in r] + [depot]
        total += inst.route_fixed_cost
        total += sum(edge_cost(inst, nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))
    return total


def solve_depot_cvrp_pyvrp(inst, depot: int, customers: list[int], num_vehicles: int,
                            time_limit_s: float, seed: int):
    """CVRP de un solo deposito con PyVRP normal (sin costos de deposito --
    ya estan fijos y se suman aparte). Devuelve (routes, cost) o (None, inf)."""
    m = inst.m
    locations = [depot] + [m + c for c in customers]  # 0 = deposito (indice local)
    n = len(locations)

    dist = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = int(round(edge_cost(inst, locations[i], locations[j]) * SCALE))

    clients = [
        pyvrp.Client(inst.coords[m + c][0], inst.coords[m + c][1],
                     delivery=[inst.demands[c]], name=str(c))
        for c in customers
    ]
    depot_obj = pyvrp.Depot(inst.coords[depot][0], inst.coords[depot][1], name="depot")
    fixed = int(round(inst.route_fixed_cost * SCALE))
    vehicle_type = pyvrp.VehicleType(
        num_available=num_vehicles,
        capacity=[inst.vehicle_capacity],
        start_depot=0,
        end_depot=0,
        fixed_cost=fixed,
        name="veh",
    )

    data = pyvrp.ProblemData(clients, [depot_obj], [vehicle_type], [dist], [np.zeros_like(dist)])
    stop = pyvrp.stop.MaxRuntime(time_limit_s)
    result = pyvrp.solve(data, stop=stop, seed=seed)

    if not result.is_feasible():
        return None, float("inf")

    routes = []
    for route in result.best.routes():
        # route.visits() son indices 1..len(customers) (0 es el deposito)
        cust_idxs = [customers[v - 1] for v in route.visits()]
        if cust_idxs:
            routes.append(cust_idxs)

    return routes, result.cost() / SCALE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=Path)
    ap.add_argument("input_sol", type=Path)
    ap.add_argument("output_sol", type=Path)
    ap.add_argument("--time-per-depot", type=float, default=30.0)
    ap.add_argument("--vehicle-buffer", type=int, default=3,
                     help="vehiculos extra permitidos mas alla del numero de rutas de la entrada")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inst = parse_instance(args.instance)
    ensure_distance_cache(inst)
    original_routes = parse_sol_routes(args.input_sol, inst.m)

    routes_by_depot: dict[int, list[list[int]]] = {}
    t0 = time.time()
    for depot, orig_routes in sorted(original_routes.items()):
        if not orig_routes:
            continue
        customers = [c for r in orig_routes for c in r]
        orig_cost = depot_route_cost(inst, depot, orig_routes)
        current_route_count = len(orig_routes)
        num_vehicles = min(inst.depot_max_vehicles[depot], current_route_count + args.vehicle_buffer)

        new_routes, new_cost = solve_depot_cvrp_pyvrp(
            inst, depot, customers, num_vehicles, args.time_per_depot, args.seed
        )

        if new_routes is not None and new_cost < orig_cost - 1e-6:
            routes_by_depot[depot] = new_routes
            print(f"  deposito {depot+1}: {len(customers)} clientes, {orig_cost:.2f} -> {new_cost:.2f} "
                  f"(transcurrido {time.time()-t0:.0f}s)", file=sys.stderr)
        else:
            routes_by_depot[depot] = orig_routes
            reason = "sin solucion" if new_routes is None else f"sin mejora ({new_cost:.2f})"
            print(f"  deposito {depot+1}: se conserva original ({orig_cost:.2f}), PyVRP {reason} "
                  f"(transcurrido {time.time()-t0:.0f}s)", file=sys.stderr)

    args.output_sol.parent.mkdir(parents=True, exist_ok=True)
    cost = write_solution(inst, routes_by_depot, args.output_sol)
    print(f"{inst.name}: costo reoptimizado={cost:.2f}")

    result = verify_solution(inst, args.output_sol)
    print(f"feasible={result.feasible} recomputed_cost={result.recomputed_cost:.2f}")
    for e in result.errors:
        print("  ERROR:", e)


if __name__ == "__main__":
    main()
