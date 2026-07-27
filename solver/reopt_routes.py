#!/usr/bin/env python3
"""Re-optimize the ROUTES of an existing feasible .sol.txt using Google
OR-Tools' CVRP solver, keeping the depot-opening / customer-to-depot
assignment fixed exactly as it is in the input solution.

Why this is safe: which depots are open and which customers each one
serves is the hard combinatorial decision (facility location); once that's
fixed, splitting one depot's customer set into vehicle routes is a plain
single-depot CVRP, and OR-Tools' routing solver (guided local search,
Lin-Kernighan-style moves) is a much stronger CVRP solver than the
hand-rolled FFD+nearest-neighbour+2-opt+relocate/swap in clrp_solver.py.
Since the customer set per depot doesn't change, depot-capacity and
customer-coverage feasibility carry over automatically from the input
solution -- only route count/order can change, and only if it lowers cost.

Usage:
    python solver/reopt_routes.py <instance.txt> <input.sol.txt> <output.sol.txt> [--time-per-depot 30]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parent.parent / "smio_clrp_verify_src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from clrp_solver import parse_instance, edge_cost, ensure_distance_cache, compute_total_cost, write_solution
from smio_clrp_verify.verify import verify_solution

SCALE = 10  # our distances/costs have 1 decimal place; scale to integers for OR-Tools


def parse_sol_routes(sol_path: Path, m: int) -> dict[int, list[list[int]]]:
    """Return {depot_idx(0-based): [[customer_idx(0-based), ...], ...]} from a
    .sol.txt, preserving the existing per-route split -- this is the
    guaranteed-feasible fallback if OR-Tools can't beat it (or can't find
    anything at all) for some depot."""
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


def solve_depot_cvrp(inst, depot: int, customers: list[int], num_vehicles: int, time_limit_s: float):
    """Solve the single-depot CVRP for `customers` assigned to `depot`.
    Returns (routes, cost) with routes = list of list of 0-based customer
    indices, cost = route_fixed_cost*used_vehicles + total distance (NOT
    including the depot's own opening cost -- that doesn't depend on routing).
    """
    m = inst.m
    nodes = [depot] + [m + c for c in customers]  # node 0 = depot in this local model
    n = len(nodes)
    demands = [0] + [inst.demands[c] for c in customers]
    Q = inst.vehicle_capacity

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        a = nodes[manager.IndexToNode(from_index)]
        b = nodes[manager.IndexToNode(to_index)]
        return int(round(edge_cost(inst, a, b) * SCALE))

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(index):
        return demands[manager.IndexToNode(index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [Q] * num_vehicles, True, "Capacity")

    fixed = int(round(inst.route_fixed_cost * SCALE))
    for v in range(num_vehicles):
        routing.SetFixedCostOfVehicle(fixed, v)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(math.ceil(time_limit_s)))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None, math.inf

    routes = []
    total_scaled = 0
    for v in range(num_vehicles):
        index = routing.Start(v)
        route = []
        route_scaled = 0
        while not routing.IsEnd(index):
            node = nodes[manager.IndexToNode(index)]
            nxt = solution.Value(routing.NextVar(index))
            if manager.IndexToNode(index) != 0:
                route.append(node - m)
            route_scaled += distance_callback(index, nxt)
            index = nxt
        if route:
            routes.append(route)
            total_scaled += route_scaled + fixed
    return routes, total_scaled / SCALE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=Path)
    ap.add_argument("input_sol", type=Path)
    ap.add_argument("output_sol", type=Path)
    ap.add_argument("--time-per-depot", type=float, default=30.0)
    ap.add_argument("--vehicle-buffer", type=int, default=3,
                     help="extra vehicles allowed beyond the input solution's route count for that depot")
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
        new_routes, new_cost = solve_depot_cvrp(inst, depot, customers, num_vehicles, args.time_per_depot)

        # Siempre se compara contra la solucion original y se usa la que
        # sea mas barata -- si OR-Tools no encuentra nada factible, o su
        # resultado sale peor, la solucion original ya probada factible
        # nunca se descarta (ningun cliente puede quedar sin cubrir).
        if new_routes is not None and new_cost < orig_cost - 1e-6:
            routes_by_depot[depot] = new_routes
            print(f"  depot {depot+1}: {len(customers)} customers, {orig_cost:.2f} -> {new_cost:.2f} "
                  f"(elapsed {time.time()-t0:.0f}s)", file=sys.stderr)
        else:
            routes_by_depot[depot] = orig_routes
            reason = "no solution" if new_routes is None else f"no improvement ({new_cost:.2f})"
            print(f"  depot {depot+1}: kept original ({orig_cost:.2f}), OR-Tools {reason} "
                  f"(elapsed {time.time()-t0:.0f}s)", file=sys.stderr)

    cost = write_solution(inst, routes_by_depot, args.output_sol)
    print(f"{inst.name}: reoptimized cost={cost:.2f}")

    result = verify_solution(inst, args.output_sol)
    print(f"feasible={result.feasible} recomputed_cost={result.recomputed_cost:.2f}")
    for e in result.errors:
        print("  ERROR:", e)


if __name__ == "__main__":
    main()
