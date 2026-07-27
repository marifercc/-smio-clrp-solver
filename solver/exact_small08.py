#!/usr/bin/env python3
"""Exact-ish check for clrp-small-08 using OR-Tools routing.

Every heuristic run so far (60+ seeds, two different local-search variants)
converges to the exact same 3 open depots (ids 2, 3, 4) and a best cost of
8780.77 -- never any other depot combination. That strongly suggests those
3 depots are the right structural choice and 8780.77 might already be at or
very near the true optimum for "given these 3 depots, what's the cheapest
way to serve all 80 customers".

This script fixes the depot-opening decision (2, 3, 4 -- confirmed above)
and hands the remaining problem -- which depot serves which customer, and
how the routes are sequenced -- to OR-Tools' routing solver (a C++ engine,
much faster per move than our pure-Python local search), to see whether it
can find anything below the routing-cost we've already reached.

Usage:
    python solver/exact_small08.py [--time-limit 120]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parent.parent / "smio_clrp_verify_src"
sys.path.insert(0, str(_REPO_SRC))
from smio_clrp_verify.verify import parse_instance  # noqa: E402

from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # noqa: E402


OPEN_DEPOTS_1BASED = [2, 3, 4]  # confirmed by 60+ heuristic runs -- see docstring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=int, default=120)
    args = ap.parse_args()

    inst = parse_instance("Instancias_oficiales/clrp-small-08.txt")
    m, n = inst.m, inst.n
    Q = inst.vehicle_capacity
    g = inst.route_fixed_cost

    open_depots_0based = [d - 1 for d in OPEN_DEPOTS_1BASED]
    opening_cost_fixed = sum(inst.opening_costs[d] for d in open_depots_0based)
    vehicles_per_depot = [inst.depot_max_vehicles[d] for d in open_depots_0based]

    # Nodo 0 = deposito "comodin" que no se usa como inicio/fin real de
    # ningun vehiculo (OR-Tools exige que todo vehiculo tenga inicio/fin
    # validos, pero como fijamos SOLO los 3 depositos abiertos, usamos sus
    # propios indices originales como start/end de cada bloque de vehiculos).
    num_locations = m + n
    manager_starts = []
    manager_ends = []
    for d, v in zip(open_depots_0based, vehicles_per_depot):
        manager_starts += [d] * v
        manager_ends += [d] * v
    num_vehicles = len(manager_starts)

    def euclid(a: int, b: int) -> float:
        ax, ay = inst.coords[a]
        bx, by = inst.coords[b]
        return round(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2), 1)

    # OR-Tools quiere costos enteros -- escalamos x10 (nuestras distancias ya
    # vienen redondeadas a 1 decimal) y desescalamos al final.
    SCALE = 10

    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, manager_starts, manager_ends)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        a = manager.IndexToNode(from_index)
        b = manager.IndexToNode(to_index)
        return int(round(euclid(a, b) * SCALE))

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return inst.demands[node - m] if node >= m else 0

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [Q] * num_vehicles, True, "Capacity")

    for v in range(num_vehicles):
        routing.SetFixedCostOfVehicle(int(round(g * SCALE)), v)

    # Los 5 depositos CERRADOS no deben aparecer como nodos visitables (no
    # tienen clientes ni son depositos abiertos) -- como nunca se agregan
    # como start/end y ningun cliente los referencia, OR-Tools simplemente
    # no los toca; no hace falta excluirlos aparte.

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromSeconds(args.time_limit)

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        print("OR-Tools no encontro ninguna solucion factible.")
        return

    total_distance = 0.0
    routes_used = 0
    for v in range(num_vehicles):
        index = routing.Start(v)
        route_nodes = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            index = solution.Value(routing.NextVar(index))
        if len(route_nodes) > 1:  # mas que solo el deposito -- vehiculo usado
            routes_used += 1
            nodes = route_nodes + [manager.IndexToNode(index)]
            for i in range(len(nodes) - 1):
                total_distance += euclid(nodes[i], nodes[i + 1])

    total_cost = opening_cost_fixed + g * routes_used + total_distance
    print(f"OR-Tools (time_limit={args.time_limit}s): rutas usadas={routes_used}, "
          f"distancia={total_distance:.2f}, apertura={opening_cost_fixed:.2f}, "
          f"COSTO TOTAL={total_cost:.2f}")
    print("Comparar contra: nuestro mejor heuristico = 8780.77, objetivo SEIN3 = 8779.87")


if __name__ == "__main__":
    main()
