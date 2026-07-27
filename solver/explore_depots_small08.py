#!/usr/bin/env python3
"""Every heuristic run on clrp-small-08 (60+ seeds, two algorithms, plus a
quick OR-Tools attempt) converged to the SAME 3 open depots: {2, 3, 4}
(1-based ids). That's suspicious -- it might mean the greedy facility
location step is structurally biased toward that combination and never
even tries the alternatives. Since there are only 5 depots total, there are
just 2**5 - 1 = 31 possible non-empty subsets to open. This script brute
forces ALL of them: for each feasible subset, force it open, build a quick
initial solution (no expensive LNS yet), and rank by cost. If {2,3,4}
really is the best structural choice, it should come out on top here too;
if something else scores better, THAT'S the one worth throwing our full
local search at instead.

Usage:
    python solver/explore_depots_small08.py
"""
from __future__ import annotations

import itertools
import signal
import sys
from pathlib import Path


class _TimedOut(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimedOut()

_REPO_SRC = Path(__file__).resolve().parent.parent / "smio_clrp_verify_src"
sys.path.insert(0, str(_REPO_SRC))
from smio_clrp_verify.verify import parse_instance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clrp_solver import (  # noqa: E402
    ensure_distance_cache, edge_cost, pack_routes, order_route, compute_total_cost,
)


def build_forced_assignment(inst, open_depots: list[int]):
    """Greedy nearest-depot-with-capacity assignment, restricted to a FIXED
    set of already-open depots (no facility-location decision-making --
    that part is what we're testing here)."""
    m, n = inst.m, inst.n
    eff_capacity = {d: min(inst.depot_capacities[d], inst.depot_max_vehicles[d] * inst.vehicle_capacity)
                    for d in open_depots}
    assignment = {d: [] for d in open_depots}
    load = {d: 0 for d in open_depots}

    order = sorted(range(n), key=lambda c: -inst.demands[c])
    for c in order:
        dem = inst.demands[c]
        candidates = sorted(open_depots, key=lambda d: edge_cost(inst, d, m + c))
        placed = False
        for d in candidates:
            if load[d] + dem <= eff_capacity[d]:
                assignment[d].append(c)
                load[d] += dem
                placed = True
                break
        if not placed:
            # ninguno de los depositos abiertos tiene lugar -- este
            # subconjunto no es factible para esta instancia
            return None
    return assignment


def quick_cost(inst, open_depots: list[int]) -> float | None:
    assignment = build_forced_assignment(inst, open_depots)
    if assignment is None:
        return None
    routes_by_depot = {}
    for d, custs in assignment.items():
        if not custs:
            continue
        bins = pack_routes(inst, d, custs)  # strict=False fallback if needed
        ordered = [order_route(inst, d, b) for b in bins]
        routes_by_depot[d] = [r for r in ordered if r]
    return compute_total_cost(inst, routes_by_depot)


def main():
    inst = parse_instance("Instancias_oficiales/clrp-small-08.txt")
    ensure_distance_cache(inst)
    m = inst.m
    total_demand = sum(inst.demands)

    results = []
    for r in range(1, m + 1):
        for combo in itertools.combinations(range(m), r):
            total_cap = sum(min(inst.depot_capacities[d], inst.depot_max_vehicles[d] * inst.vehicle_capacity)
                             for d in combo)
            if total_cap < total_demand:
                continue  # ni de milagro alcanza la capacidad -- ni lo intentes
            # Combos con pocos depositos abiertos pueden dejar demanda MUY
            # ajustada contra la capacidad de vehiculos disponible, lo que
            # hace que la reparacion de bin-packing (pack_routes) intente
            # fusionar bins una y otra vez sin nunca lograrlo -- un guardia
            # de tiempo por combo evita que uno solo trabe todo el barrido.
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(5)
            try:
                cost = quick_cost(inst, list(combo))
            except _TimedOut:
                cost = None
                print(f"  (combo {[d + 1 for d in combo]} tardo demasiado, se salta)")
            finally:
                signal.alarm(0)
            if cost is not None:
                ids_1based = [d + 1 for d in combo]
                results.append((cost, ids_1based))

    results.sort(key=lambda t: t[0])
    print(f"{len(results)} subconjuntos de depositos factibles evaluados (de 31 posibles).")
    print("Top 8 (costo de construccion rapida, SIN busqueda local todavia):")
    for cost, ids in results[:8]:
        marker = "  <-- la que siempre usa el heuristico" if ids == [2, 3, 4] else ""
        print(f"  depositos {ids}: costo={cost:.2f}{marker}")


if __name__ == "__main__":
    main()
