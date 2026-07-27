#!/usr/bin/env python3
"""Joint customer-reassignment + routing re-optimization via OR-Tools,
restricted to the depots already open in an input solution (the facility-
location decision itself is left alone -- extensively tested separately;
this only asks "given these open depots, is there a better way to split
customers and route them than what the heuristic found").

Unlike reopt_routes.py (which fixes each depot's customer set and only
re-splits/re-routes within it), this lets OR-Tools move customers BETWEEN
the open depots too. That requires enforcing each depot's true aggregate
capacity across all of its vehicles (W_d), which OR-Tools' per-vehicle
capacity dimension does not do by itself (it only caps each vehicle's own
load at Q). The fix: add one linear constraint per depot, on top of the
normal per-vehicle Capacity dimension, that sums the end-of-route load
(CumulVar at each vehicle's End index) across all vehicles belonging to
that depot and bounds it by W_d -- this is the standard OR-Tools pattern
for a shared/aggregate resource limit across a vehicle group.

Falls back to the original per-depot routes (guaranteed feasible) if
OR-Tools can't find anything, or if what it finds doesn't beat the
original cost -- so this can only ever match or improve the input, never
silently lose customer coverage or regress.

Usage:
    python solver/joint_reopt2.py <instance.txt> <input.sol.txt> <output.sol.txt> [--time-limit 300] [--vehicle-buffer 3]
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
from reopt_routes import parse_sol_routes, depot_route_cost
from smio_clrp_verify.verify import verify_solution

SCALE = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=Path)
    ap.add_argument("input_sol", type=Path)
    ap.add_argument("output_sol", type=Path)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--vehicle-buffer", type=int, default=3)
    args = ap.parse_args()

    inst = parse_instance(args.instance)
    ensure_distance_cache(inst)
    m, Q = inst.m, inst.vehicle_capacity

    original_routes = parse_sol_routes(args.input_sol, m)
    open_depots = sorted(d for d, rs in original_routes.items() if rs)
    original_cost = sum(depot_route_cost(inst, d, original_routes[d]) for d in open_depots)
    original_cost += sum(inst.opening_costs[d] for d in open_depots)
    print(f"open depots: {[d+1 for d in open_depots]}, original cost={original_cost:.2f}", file=sys.stderr)

    starts, ends, vehicle_depot = [], [], []
    for d in open_depots:
        current_route_count = len(original_routes[d])
        num_v = min(inst.depot_max_vehicles[d], current_route_count + args.vehicle_buffer)
        for _ in range(num_v):
            starts.append(d)
            ends.append(d)
            vehicle_depot.append(d)
    num_vehicles = len(starts)
    print(f"total vehicles in joint model: {num_vehicles}", file=sys.stderr)

    all_customers = sorted({c for rs in original_routes.values() for r in rs for c in r})
    # Local node numbering: 0..m-1 are the open depots (by global id), then
    # customers. We keep depots at their GLOBAL id as the node label (0..m-1
    # are unused for depots not in open_depots, but RoutingIndexManager needs
    # a dense 0..total_nodes-1 space) -- simplest: node i in [0,m) maps to
    # global depot i directly (all m depots exist as nodes, unused ones just
    # never get a vehicle), then customers follow at m + local customer idx.
    manager = pywrapcp.RoutingIndexManager(m + len(all_customers), num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    def global_node(local_node: int) -> int:
        return local_node if local_node < m else m + all_customers[local_node - m]

    def distance_callback(from_index, to_index):
        a = global_node(manager.IndexToNode(from_index))
        b = global_node(manager.IndexToNode(to_index))
        return int(round(edge_cost(inst, a, b) * SCALE))

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(index):
        local_node = manager.IndexToNode(index)
        if local_node < m:
            return 0
        return inst.demands[all_customers[local_node - m]]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [Q] * num_vehicles, True, "Capacity")
    capacity_dim = routing.GetDimensionOrDie("Capacity")

    # Costo fijo aproximado por vehiculo (route_fixed_cost + una fraccion del
    # costo de apertura de su deposito) -- solo para GUIAR la busqueda hacia
    # soluciones con pocos depositos/rutas; el costo real que decide si se
    # acepta el resultado se recalcula despues con compute_total_cost.
    for v in range(num_vehicles):
        d = vehicle_depot[v]
        n_v_this_depot = sum(1 for vv in vehicle_depot if vv == d)
        fixed = inst.route_fixed_cost + inst.opening_costs[d] / n_v_this_depot
        routing.SetFixedCostOfVehicle(int(round(fixed * SCALE)), v)

    # Restriccion de capacidad AGREGADA por deposito: la carga total de
    # TODOS sus vehiculos (no cada uno por separado) no puede pasar W_d.
    solver = routing.solver()
    for d in open_depots:
        vehicle_indices = [v for v, vd in enumerate(vehicle_depot) if vd == d]
        end_cumuls = [capacity_dim.CumulVar(routing.End(v)) for v in vehicle_indices]
        solver.Add(solver.Sum(end_cumuls) <= int(inst.depot_capacities[d]))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(math.ceil(args.time_limit)))

    solution = routing.SolveWithParameters(params)
    routes_by_depot: dict[int, list[list[int]]] = {}
    if solution is not None:
        for v in range(num_vehicles):
            d = vehicle_depot[v]
            index = routing.Start(v)
            route = []
            while not routing.IsEnd(index):
                local_node = manager.IndexToNode(index)
                if local_node >= m:
                    route.append(all_customers[local_node - m])
                index = solution.Value(routing.NextVar(index))
            if route:
                routes_by_depot.setdefault(d, []).append(route)

    new_cost = None
    if solution is not None and routes_by_depot:
        new_cost = compute_total_cost(inst, routes_by_depot)
        print(f"joint model found cost={new_cost:.2f}", file=sys.stderr)

    if new_cost is not None and new_cost < original_cost - 1e-6:
        final_routes = routes_by_depot
        print("KEEPING joint-reoptimized solution (improved)", file=sys.stderr)
    else:
        final_routes = original_routes
        print(f"KEEPING original solution (joint reopt {'found nothing' if new_cost is None else f'no improvement ({new_cost:.2f})'})",
              file=sys.stderr)

    cost = write_solution(inst, final_routes, args.output_sol)
    print(f"{inst.name}: final cost={cost:.2f}")
    result = verify_solution(inst, args.output_sol)
    print(f"feasible={result.feasible} recomputed_cost={result.recomputed_cost:.2f}")
    for e in result.errors:
        print("  ERROR:", e)


if __name__ == "__main__":
    main()
