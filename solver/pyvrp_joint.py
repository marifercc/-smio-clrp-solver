#!/usr/bin/env python3
"""Reoptimiza TODOS los depositos abiertos de una .sol.txt EN CONJUNTO,
dejando que PyVRP reasigne libremente que cliente sirve cada deposito (un
nivel mas agresivo que pyvrp_route.py, que solo reordena rutas DENTRO de
cada deposito sin mover clientes entre ellos). Solo se fija CUALES depositos
estan abiertos -- la asignacion cliente-deposito y el ruteo completo se
resuelven juntos con el motor HGS de PyVRP.

Precaucion: PyVRP modela la capacidad por VEHICULO (Q por ruta), no una
capacidad agregada por deposito (W_i). Si W_i es mas estricto que
max_vehicles[d]*Q para algun deposito, PyVRP podria proponer una solucion
que "parezca" factible para el pero viole W_i sin darse cuenta. Por eso este
script SIEMPRE verifica con el verificador OFICIAL antes de aceptar el
resultado -- si sale infactible o no mejora, se descarta y se conserva la
solucion de entrada tal cual (que ya se asume factible).

Uso:
    python solver/pyvrp_joint.py <instancia.txt> <entrada.sol.txt> <salida.sol.txt> [--time-limit 120] [--seed 0]
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

from clrp_solver import parse_instance, ensure_distance_cache, edge_cost, write_solution, compute_total_cost
from smio_clrp_verify.verify import verify_solution

SCALE = 10


def parse_sol_routes(sol_path: Path, m: int) -> dict[int, list[list[int]]]:
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


def solve_joint(inst, open_depots: list[int], customers: list[int], time_limit_s: float, seed: int):
    """Resuelve el VRP multi-deposito completo (todos los depositos abiertos
    a la vez, clientes libres de moverse entre ellos). Devuelve
    routes_by_depot o None si PyVRP no encuentra nada factible segun su
    propio modelo (capacidad por vehiculo)."""
    m = inst.m
    dep_locs = list(open_depots)
    cust_locs = [m + c for c in customers]
    all_locs = dep_locs + cust_locs
    n = len(all_locs)
    n_dep = len(dep_locs)

    dist = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = int(round(edge_cost(inst, all_locs[i], all_locs[j]) * SCALE))

    clients = [
        pyvrp.Client(inst.coords[m + c][0], inst.coords[m + c][1],
                     delivery=[inst.demands[c]], name=str(c))
        for c in customers
    ]
    depot_objs = [
        pyvrp.Depot(inst.coords[d][0], inst.coords[d][1], name=str(d))
        for d in open_depots
    ]

    fixed = int(round(inst.route_fixed_cost * SCALE))
    # PyVRP modela la capacidad POR VEHICULO (Q por ruta), no una capacidad
    # agregada por deposito (W_i). Para que nunca pueda violar W_i sin
    # querer, topamos el numero de vehiculos disponibles en cada deposito de
    # forma que vehiculos*Q <= W_i siempre -- asi, sin importar como PyVRP
    # arme las rutas, la demanda total que le cae a ese deposito jamas puede
    # pasarse de su capacidad real.
    Q = inst.vehicle_capacity
    vehicle_types = [
        pyvrp.VehicleType(
            num_available=min(inst.depot_max_vehicles[d], max(1, inst.depot_capacities[d] // Q)),
            capacity=[Q],
            start_depot=idx,
            end_depot=idx,
            fixed_cost=fixed,
            name=str(d),
        )
        for idx, d in enumerate(open_depots)
    ]

    data = pyvrp.ProblemData(clients, depot_objs, vehicle_types, [dist], [np.zeros_like(dist)])
    stop = pyvrp.stop.MaxRuntime(time_limit_s)
    result = pyvrp.solve(data, stop=stop, seed=seed)

    if not result.is_feasible():
        return None

    routes_by_depot: dict[int, list[list[int]]] = {}
    for route in result.best.routes():
        depot_idx = route.start_depot()
        orig_depot = open_depots[depot_idx]
        cust_idxs = [customers[v - n_dep] for v in route.visits()]
        if cust_idxs:
            routes_by_depot.setdefault(orig_depot, []).append(cust_idxs)

    return routes_by_depot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=Path)
    ap.add_argument("input_sol", type=Path)
    ap.add_argument("output_sol", type=Path)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depots", type=str, default=None,
                     help="Forzar esta combinacion de depositos en vez de la de input_sol, "
                          "ej. '1,3,4,5,7,8,10' (1-based). input_sol solo se usa para saber "
                          "cual es el costo de referencia a superar (si esos depositos no "
                          "coinciden, la comparacion es solo informativa).")
    args = ap.parse_args()

    inst = parse_instance(args.instance)
    ensure_distance_cache(inst)
    original_routes = parse_sol_routes(args.input_sol, inst.m)

    original_cost = compute_total_cost(inst, original_routes)

    if args.depots:
        open_depots = sorted(int(x) - 1 for x in args.depots.split(","))
        customers = list(range(inst.n))
        print(f"Forzando depositos {[d + 1 for d in open_depots]} (referencia a superar: {original_cost:.2f})")
    else:
        open_depots = sorted(original_routes.keys())
        customers = [c for routes in original_routes.values() for r in routes for c in r]

    t0 = time.time()
    new_routes = solve_joint(inst, open_depots, customers, args.time_limit, args.seed)
    new_cost = compute_total_cost(inst, new_routes) if new_routes is not None else float("inf")
    elapsed = time.time() - t0

    if new_routes is not None and new_cost < original_cost - 1e-6:
        final_routes = new_routes
        print(f"mejora: {original_cost:.2f} -> {new_cost:.2f} (tiempo {elapsed:.0f}s)")
    else:
        final_routes = original_routes
        reason = "infactible segun PyVRP" if new_routes is None else f"sin mejora ({new_cost:.2f})"
        print(f"sin mejora, se conserva original ({original_cost:.2f}) -- PyVRP {reason} "
              f"(tiempo {elapsed:.0f}s)")

    args.output_sol.parent.mkdir(parents=True, exist_ok=True)
    cost = write_solution(inst, final_routes, args.output_sol)
    print(f"{inst.name}: costo final escrito={cost:.2f}")

    result = verify_solution(inst, args.output_sol)
    print(f"verificador oficial: feasible={result.feasible} recomputed_cost={result.recomputed_cost:.2f}")
    for e in result.errors:
        print("  ERROR:", e)

    if not result.feasible:
        # Salvavidas: el resultado conjunto paso el chequeo de PyVRP pero NO
        # el verificador oficial (probablemente por la capacidad agregada de
        # deposito, W_i, que PyVRP no modela) -- volvemos a la original.
        print("El verificador oficial marco infactible -- reescribiendo con la solucion original.")
        cost = write_solution(inst, original_routes, args.output_sol)
        result = verify_solution(inst, args.output_sol)
        print(f"{inst.name}: costo final (fallback)={cost:.2f} feasible={result.feasible}")


if __name__ == "__main__":
    main()
