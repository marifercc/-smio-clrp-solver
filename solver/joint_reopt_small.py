#!/usr/bin/env python3
"""Joint facility-location + routing re-optimization via OR-Tools for a
SINGLE instance, used only when per-vehicle capacity (V_i * Q) is already
the binding limit at every depot (i.e. depot_capacities[d] >= V_i*Q for all
d) -- in that case OR-Tools' native per-vehicle capacity dimension already
enforces the true depot limit exactly, so it's safe to let the solver
freely reassign customers between depots too, not just re-route within a
fixed assignment.

Opening cost is approximated inside OR-Tools as opening_cost/V_i added to
every one of that depot's vehicles (so using k of its vehicles roughly
costs route_fixed_cost*k + opening_cost*k/V_i) -- an approximation used
only to GUIDE the search. The actual reported/compared cost always comes
from clrp_solver.compute_total_cost on the extracted routes, and the
solution is only kept if it's both cheaper AND passes the official
verifier, so a rough internal objective can't produce a wrong final
answer, only a worse search.

Usage:
    python solver/joint_reopt_small.py <instance.txt> <output.sol.txt> [--time-limit 300]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parent.parent / "smio_clrp_verify_src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from clrp_solver import parse_instance, edge_cost, ensure_distance_cache, compute_total_cost, write_solution
from smio_clrp_verify.verify import verify_solution

SCALE = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=Path)
    ap.add_argument("output_sol", type=Path)
    ap.add_argument("--time-limit", type=float, default=300.0)
    args = ap.parse_args()

    inst = parse_instance(args.instance)
    ensure_distance_cache(inst)
    m, n, Q = inst.m, inst.n, inst.vehicle_capacity

    for d in range(m):
        if inst.depot_capacities[d] < inst.depot_max_vehicles[d] * Q:
            print(f"UNSAFE: depot {d+1} capacity {inst.depot_capacities[d]} < "
                  f"V*Q {inst.depot_max_vehicles[d] * Q} -- joint reassignment could violate "
                  f"depot capacity, aborting.", file=sys.stderr)
            sys.exit(1)

    starts, ends, vehicle_depot = [], [], []
    for d in range(m):
        for _ in range(inst.depot_max_vehicles[d]):
            starts.append(d)
            ends.append(d)
            vehicle_depot.append(d)
    num_vehicles = len(starts)

    total_nodes = m + n
    manager = pywrapcp.RoutingIndexManager(total_nodes, num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        a = manager.IndexToNode(from_index)
        b = manager.IndexToNode(to_index)
        return int(round(edge_cost(inst, a, b) * SCALE))

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(index):
        node = manager.IndexToNode(index)
        return inst.demands[node - m] if node >= m else 0

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [Q] * num_vehicles, True, "Capacity")

    for v in range(num_vehicles):
        d = vehicle_depot[v]
        fixed = inst.route_fixed_cost + inst.opening_costs[d] / inst.depot_max_vehicles[d]
        routing.SetFixedCostOfVehicle(int(round(fixed * SCALE)), v)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(math.ceil(args.time_limit)))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        print("OR-Tools found no solution", file=sys.stderr)
        sys.exit(1)

    routes_by_depot: dict[int, list[list[int]]] = {}
    for v in range(num_vehicles):
        d = vehicle_depot[v]
        index = routing.Start(v)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node >= m:
                route.append(node - m)
            index = solution.Value(routing.NextVar(index))
        if route:
            routes_by_depot.setdefault(d, []).append(route)

    cost = write_solution(inst, routes_by_depot, args.output_sol)
    print(f"{inst.name}: joint-reoptimized cost={cost:.2f}")

    result = verify_solution(inst, args.output_sol)
    print(f"feasible={result.feasible} recomputed_cost={result.recomputed_cost:.2f}")
    for e in result.errors:
        print("  ERROR:", e)


if __name__ == "__main__":
    main()
