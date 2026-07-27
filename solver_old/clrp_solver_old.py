#!/usr/bin/env python3
"""Baseline constructive heuristic for the SMIO-Hexaly CLRP Challenge 2026.

Pipeline:
  1. Parse instance with the official verifier's parser (smio_clrp_verify.verify).
  2. Greedy capacitated facility-location: open depots one at a time, always
     picking the depot whose opening produces the largest net cost reduction
     (reassigning nearby customers to it), until every customer can be served
     within depot capacity AND depot vehicle-count limits (effective capacity
     = min(W_i, V_i * Q)). This is the classic "greedy add" heuristic for
     capacitated facility location.
  3. Per open depot: split its assigned customers into vehicle routes with
     First-Fit-Decreasing bin packing on demand (capacity Q), then order each
     route with nearest-neighbour + 2-opt.
  4. Write a .sol file in the official format and (optionally) verify it with
     the official verifier.

Usage:
    python clrp_solver.py <instance.txt> <output.sol.txt> [--verify]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

# Make the bundled generator/verifier repo importable without pip install
# (its pyproject.toml pins python>=3.11; the code itself runs fine on 3.10).
_REPO_SRC = Path(__file__).resolve().parent.parent / "smio_clrp_verify_src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import numpy as np

from smio_clrp_verify.verify import parse_instance, verify_solution  # noqa: E402


# ---------------------------------------------------------------------------
# Distance helper (mirrors smio_clrp_verify.verify._edge_cost exactly)
# ---------------------------------------------------------------------------

def edge_cost(inst, a: int, b: int) -> float:
    """0-based node indices a, b (0..m-1 depots, m..m+n-1 customers)."""
    if inst.distance_format == "FULL_MATRIX":
        return float(inst.distance_matrix[a, b])
    ax, ay = inst.coords[a]
    bx, by = inst.coords[b]
    dx, dy = ax - bx, ay - by
    return round(math.sqrt(dx * dx + dy * dy), 1)


# ---------------------------------------------------------------------------
# Step 1: capacitated facility location (greedy add)
# ---------------------------------------------------------------------------

def solve_facility_location(inst) -> dict[int, list[int]]:
    """Return {depot_idx(0-based): [customer_idx(0-based), ...]}.

    Algoritmo "greedy add": arranca sin ningun deposito abierto y, en cada
    vuelta, evalua TODOS los depositos aun cerrados como candidatos a abrir.
    Para cada candidato d, mira que clientes convendria servir desde d (los
    que hoy no tienen deposito, o para los que d queda mas cerca que su
    deposito actual), llena d hasta su capacidad efectiva empezando por los
    clientes mas baratos de servir, y calcula la "ganancia neta" de abrirlo:
    (distancia que se ahorra reasignando esos clientes) - (costo de apertura
    del deposito). Se abre el deposito con mejor ganancia y se repite, hasta
    que abrir cualquier otro deposito ya no compensa su costo de apertura
    (o hasta que todos los clientes quedan servidos, si ninguno compensa
    pero todavia faltan clientes por asignar -- ahi se fuerza la apertura
    igual, porque la solucion tiene que ser factible).
    """
    m, n = inst.m, inst.n
    # "Capacidad efectiva" de un deposito: no sirve de nada que W_i (limite de
    # carga) sea mayor a V_i*Q (lo que realmente pueden cargar sus vehiculos
    # combinados), asi que el limite real es el menor de los dos.
    eff_capacity = [min(inst.depot_capacities[d], inst.depot_max_vehicles[d] * inst.vehicle_capacity)
                    for d in range(m)]

    open_depots: set[int] = set()
    closed_depots: set[int] = set(range(m))
    # best_depot[c] / best_cost[c]: a que deposito esta asignado hoy cada
    # cliente y cuanto cuesta esa asignacion (-1 / inf si todavia no tiene).
    best_depot = [-1] * n
    best_cost = [math.inf] * n

    def dist_dc(d: int, c: int) -> float:
        return edge_cost(inst, d, m + c)

    while True:
        unassigned = set(c for c in range(n) if best_depot[c] == -1)
        served_ok = not unassigned

        best_choice = None  # (d, gain, forced, taken)

        for d in closed_depots:
            cap = eff_capacity[d]
            if cap <= 0:
                continue
            # Un cliente es candidato a moverse a d si todavia no tiene
            # deposito, o si d le queda mas cerca que su deposito actual.
            candidates = []
            for c in range(n):
                dc = dist_dc(d, c)
                if best_depot[c] == -1 or dc < best_cost[c]:
                    candidates.append((dc, c))
            # Prioriza clientes sin servir sobre los que solo mejorarian de
            # deposito, y entre iguales el mas barato primero (greedy knapsack).
            candidates.sort(key=lambda t: (t[1] not in unassigned, t[0]))
            taken = []
            load = 0
            for dc, c in candidates:
                dem = inst.demands[c]
                if load + dem > cap:
                    continue
                load += dem
                taken.append((c, dc))
            if not taken:
                continue
            newly_served = sum(1 for c, _ in taken if c in unassigned)
            # Cuanto se ahorra en distancia reasignando: para clientes que ya
            # tenian deposito, la diferencia entre su costo viejo y el nuevo;
            # para clientes sin servir todavia, no hay "ahorro" (no habia
            # nada antes), solo se suma su costo como parte de la apertura.
            distance_savings = sum(
                (best_cost[c] - dc) if best_depot[c] != -1 else 0.0
                for c, dc in taken
            )
            gain = distance_savings - inst.opening_costs[d]
            # "forced": este deposito rescata a algun cliente que hoy no
            # tiene a nadie que lo sirva -- se prioriza aunque su ganancia
            # neta de por si sea negativa, porque la solucion final tiene que
            # cubrir a todos los clientes.
            forced = newly_served > 0 and not served_ok
            if best_choice is None:
                best_choice = (d, gain, forced, taken)
            else:
                _, bgain, bforced, _ = best_choice
                if (forced, gain) > (bforced, bgain):
                    best_choice = (d, gain, forced, taken)

        if best_choice is None:
            if not served_ok:
                raise RuntimeError("Infeasible: total depot capacity cannot cover all demand")
            break

        d, gain, forced, taken = best_choice
        if not forced and gain <= 0 and served_ok:
            break  # abrir mas depositos ya solo sumaria costo, no compensa

        open_depots.add(d)
        closed_depots.discard(d)
        for c, dc in taken:
            best_depot[c] = d
            best_cost[c] = dc

    assignment: dict[int, list[int]] = {d: [] for d in open_depots}
    for c in range(n):
        assignment[best_depot[c]].append(c)
    return assignment


# ---------------------------------------------------------------------------
# Step 2: per-depot routing (FFD bin packing + nearest-neighbour + 2-opt)
# ---------------------------------------------------------------------------

def _ffd_pack(inst, customers: list[int], Q: int) -> tuple[list[list[int]], list[int]]:
    """First-Fit-Decreasing: ordena a los clientes de mayor a menor demanda y
    mete cada uno en el "bin" (vehiculo) ya abierto que le quede mas justo
    (menor espacio libre restante) sin pasarse de Q; si no entra en ninguno,
    abre un bin nuevo. No garantiza el numero minimo de vehiculos, pero es
    rapido y en la practica queda cerca del optimo para bin packing.
    """
    items = sorted(customers, key=lambda c: -inst.demands[c])
    bins: list[list[int]] = []
    loads: list[int] = []
    for c in items:
        dem = inst.demands[c]
        best_bin, best_slack = -1, math.inf
        for i, ld in enumerate(loads):
            if ld + dem <= Q and (Q - ld - dem) < best_slack:
                best_bin, best_slack = i, Q - ld - dem
        if best_bin >= 0:
            bins[best_bin].append(c)
            loads[best_bin] += dem
        else:
            bins.append([c])
            loads.append(dem)
    return bins, loads


def pack_routes(inst, depot: int, customers: list[int], strict: bool = False):
    """Split customers (0-based idx) assigned to `depot` into capacity-Q bins.

    If strict=True, returns None instead of a list when the customer set
    cannot be packed into depot_max_vehicles[depot] bins without exceeding Q
    (caller is expected to relocate a customer elsewhere and retry). If
    strict=False, falls back to a Q-violating forced merge as a last resort
    (used only once relocation has been exhausted).
    """
    Q = inst.vehicle_capacity
    max_v = inst.depot_max_vehicles[depot]
    bins, loads = _ffd_pack(inst, customers, Q)

    # FFD puede devolver mas bins de los V_i vehiculos disponibles. Mientras
    # sobren bins, intenta vaciar el mas liviano: si TODOS sus clientes
    # entran en algun otro bin (sin pasarse de Q), se elimina ese bin y se
    # repite. Si ningun bin se puede vaciar por completo, no hay forma de
    # bajar mas la cantidad de vehiculos con esta asignacion de clientes.
    progress = True
    while len(bins) > max_v and progress:
        progress = False
        order = sorted(range(len(bins)), key=lambda i: loads[i])
        for i in order:
            items = sorted(bins[i], key=lambda c: -inst.demands[c])
            new_loads = list(loads)
            new_bins = [list(b) for b in bins]
            ok = True
            for c in items:
                dem = inst.demands[c]
                best_j, best_slack = -1, math.inf
                for j in range(len(new_bins)):
                    if j == i:
                        continue
                    if new_loads[j] + dem <= Q and (Q - new_loads[j] - dem) < best_slack:
                        best_j, best_slack = j, Q - new_loads[j] - dem
                if best_j == -1:
                    ok = False
                    break
                new_bins[best_j].append(c)
                new_loads[best_j] += dem
            if ok:
                del new_bins[i]
                del new_loads[i]
                bins, loads = new_bins, new_loads
                progress = True
                break

    if len(bins) > max_v:
        if strict:
            return None
        print(
            f"    WARNING: depot {depot + 1} could not be packed into {max_v} "
            f"routes of capacity {Q} even after relocation -- forcing a merge "
            f"that exceeds Q; solution will be flagged infeasible.",
            file=sys.stderr,
        )
        while len(bins) > max_v:
            order = sorted(range(len(bins)), key=lambda i: loads[i])
            i, j = order[0], order[1]
            bins[j] = bins[j] + bins[i]
            loads[j] += loads[i]
            del bins[i]
            del loads[i]
    return bins


def order_route(inst, depot: int, customers: list[int]) -> list[int]:
    """Nearest-neighbour construction + 2-opt improvement for one route.

    Recibe un CONJUNTO de clientes (el orden de entrada no importa) y decide
    en que secuencia visitarlos. Primero arma una ruta golosa: desde el
    deposito, siempre salta al cliente sin visitar mas cercano. Esa ruta
    suele tener cruces innecesarios, asi que despues aplica 2-opt: prueba
    invertir cada tramo [i, j] de la ruta y se queda con la inversion si
    acorta la distancia total, repitiendo hasta que ninguna inversion ayude.
    """
    if not customers:
        return []
    remaining = set(customers)
    route = []
    current = depot
    while remaining:
        nxt = min(remaining, key=lambda c: edge_cost(inst, current, inst.m + c))
        route.append(nxt)
        remaining.discard(nxt)
        current = inst.m + nxt

    def route_len(order: list[int]) -> float:
        nodes = [depot] + [inst.m + c for c in order] + [depot]
        return sum(edge_cost(inst, nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))

    improved = True
    best_len = route_len(route)
    while improved and len(route) > 3:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                new_len = route_len(new_route)
                if new_len + 1e-9 < best_len:
                    route = new_route
                    best_len = new_len
                    improved = True
    return route


# ---------------------------------------------------------------------------
# Step 2.5: local search improvement (relocate + swap, neighbour-list pruned)
# ---------------------------------------------------------------------------

def build_neighbor_lists(inst, k: int = 15) -> list[list[int]]:
    """For each customer, its k nearest other customers (0-based indices).

    Vectorised with numpy so it stays fast even at n=3000 (large instances).
    Used to prune the relocate/swap search: we only ever try moving a
    customer into a route that already contains one of its near neighbours,
    instead of checking every route (the classic "granular" local-search
    speedup).
    """
    n, m = inst.n, inst.n and inst.m
    m = inst.m
    if inst.distance_format == "FULL_MATRIX":
        sub = inst.distance_matrix[m:m + n, m:m + n]
        lists = []
        for c in range(n):
            idx = np.argsort(sub[c])
            idx = idx[idx != c][:k]
            lists.append(idx.tolist())
        return lists
    coords = np.array(inst.coords[m:m + n], dtype=np.float64)
    lists = []
    for c in range(n):
        d = np.sqrt(((coords - coords[c]) ** 2).sum(axis=1))
        idx = np.argsort(d)
        idx = idx[idx != c][:k]
        lists.append(idx.tolist())
    return lists


def local_search(inst, routes_by_depot, neighbor_lists, time_limit: float = 30.0):
    """Improve a constructed solution with relocate + swap + depot-close moves.

    Relocate: remove a customer and re-insert it (at its cheapest position)
    into a different route, if that reduces total cost.
    Swap: exchange two customers between two routes (position-preserving),
    if that reduces total cost.
    Both moves only consider target routes/customers drawn from each
    customer's nearest-neighbour list, and both respect vehicle capacity Q
    and depot capacity W_i (the number of routes/vehicles never changes, so
    V_i is automatically respected).

    Depot-close (try_close_depot): intenta apagar un deposito entero,
    reubicando a todos sus clientes en otro lado, cuando ya no compensa
    pagar su costo de apertura (ver mas abajo).

    Estas tres jugadas se alternan en fases en vez de mezclarse en una sola
    pasada (ver el bucle principal al final de la funcion): primero
    relocate/swap corre hasta converger del todo, recien despues se intenta
    cerrar depositos, y si algo se cierra se vuelve a converger relocate/swap
    antes de probar cerrar mas. Hacerlo mezclado en una sola pasada tambien
    funciona, pero como la busqueda local es "primer movimiento que mejora"
    (no busqueda exhaustiva), el ORDEN en que se aplican las jugadas cambia a
    que optimo local se termina llegando -- cerrar un deposito a mitad de una
    pasada de relocate/swap, antes de que esa fase termine de acomodar todo,
    puede terminar en un resultado peor aunque cada jugada individual haya
    sido, en su momento, una mejora real.

    Una vez que las tres jugadas convergen a un optimo local (converge_local),
    si sobra tiempo se entra en una fase de LNS (Large Neighborhood Search):
    se destruye a proposito una porcion de la solucion (destroy: sacar un
    grupo de clientes de sus rutas) y se reconstruye (repair: reinsertarlos
    greedy), volviendo a converger relocate/swap/close sobre el resultado.
    Esto es necesario porque relocate/swap/close, al ser todas jugadas que
    solo se aceptan si mejoran el costo, nunca pueden salir por si solas de
    un optimo local -- rehacen exactamente lo mismo una y otra vez sin
    encontrar mas mejoras. La perturbacion de destroy() rompe esa parte de
    la solucion a la fuerza para que converge_local() la tenga que resolver
    de nuevo, posiblemente de otra forma. Se guarda siempre la mejor
    solucion vista (best_snapshot); si una perturbacion no mejora las cosas,
    se vuelve atras antes de probar la siguiente.

    Se detiene cuando se cumple time_limit (segundos).
    """
    Q = inst.vehicle_capacity
    m = inst.m
    depot_capacities = inst.depot_capacities

    # Representacion de trabajo: una lista plana de rutas (cada una con su
    # deposito y la lista de clientes que visita), mas indices auxiliares
    # para no tener que recorrer todo de nuevo en cada jugada:
    #   route_of[cliente]  -> indice de ruta que lo tiene asignado hoy
    #   route_load[ridx]   -> demanda total cargada en la ruta ridx
    #   depot_load[deposito] -> demanda total servida hoy por ese deposito
    routes: list[dict] = []
    for d, rlist in routes_by_depot.items():
        for r in rlist:
            routes.append({"depot": d, "custs": list(r)})

    route_of: dict[int, int] = {}
    for ridx, r in enumerate(routes):
        for c in r["custs"]:
            route_of[c] = ridx

    route_load = [sum(inst.demands[c] for c in r["custs"]) for r in routes]
    depot_load: dict[int, int] = {}
    for r in routes:
        depot_load[r["depot"]] = depot_load.get(r["depot"], 0) + sum(inst.demands[c] for c in r["custs"])

    def route_nodes(ridx: int) -> list[int]:
        """Secuencia completa de nodos de una ruta: deposito -> clientes -> deposito."""
        r = routes[ridx]
        return [r["depot"]] + [m + c for c in r["custs"]] + [r["depot"]]

    def insertion_cost(ridx: int, c: int) -> tuple[float, int]:
        """Cuanto aumenta la distancia de la ruta ridx si se inserta el
        cliente c en su mejor posicion posible (y cual es esa posicion).
        No se reordena toda la ruta -- es solo el costo marginal de "abrir
        hueco" entre dos nodos consecutivos existentes."""
        nodes = route_nodes(ridx)
        best_delta, best_pos = math.inf, -1
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i + 1]
            delta = edge_cost(inst, a, m + c) + edge_cost(inst, m + c, b) - edge_cost(inst, a, b)
            if delta < best_delta:
                best_delta, best_pos = delta, i
        return best_delta, best_pos

    def removal_saving(ridx: int, c: int) -> float:
        """Cuanto se ahorra en distancia si se saca al cliente c de la ruta
        ridx (el complemento exacto de insertion_cost)."""
        pos = routes[ridx]["custs"].index(c)
        nodes = route_nodes(ridx)
        a, b = nodes[pos], nodes[pos + 2]
        return edge_cost(inst, a, m + c) + edge_cost(inst, m + c, b) - edge_cost(inst, a, b)

    def try_close_depot(d: int) -> bool:
        """Try to shut down depot `d` by relocating every customer on its
        routes elsewhere: into an existing route at another open depot if
        one has room, or -- if none does -- into a brand-new route opened at
        an open depot that still has a free vehicle slot (V_i not yet used
        up). Commits and returns True only if the opening cost plus the
        fixed cost and distance of d's own routes exceed the total cost of
        placing every displaced customer elsewhere; otherwise leaves all
        state untouched and returns False.

        Estrategia general: primero se calcula cuanto se AHORRA si d se
        cierra (removed_cost). Despues se intenta encontrar, para cada
        cliente que quedaria sin deposito, el lugar mas barato donde
        reinsertarlo -- todo esto sobre estructuras de PRUEBA (`cand`,
        `cand_load`, `trial_depot_load`, etc.), sin tocar el estado real
        (`routes`, `route_of`, `route_load`, `depot_load`) todavia. Recien al
        final, si el costo total de reinsertar a todos (added_cost) resulta
        MENOR que lo ahorrado, se aplica todo de una vez ("commit"). Si en
        cualquier momento un cliente no tiene donde ir, o ya se ve que no
        conviene, se corta y se devuelve False sin haber tocado nada real.
        """
        route_indices = [ridx for ridx, r in enumerate(routes) if r["depot"] == d and r["custs"]]
        if not route_indices:
            return False  # d ya esta cerrado (sin rutas activas), nada que hacer

        # Costo que se recupera si d cierra: su costo de apertura, mas el
        # costo fijo y la distancia de cada una de sus rutas actuales.
        removed_cost = inst.opening_costs[d]
        for ridx in route_indices:
            removed_cost += inst.route_fixed_cost
            nodes = route_nodes(ridx)
            removed_cost += sum(edge_cost(inst, nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))

        displaced = [c for ridx in route_indices for c in routes[ridx]["custs"]]

        # Candidate targets, keyed the same way as `routes`: real route
        # indices (>=0) for existing non-empty routes at other depots, plus
        # synthetic negative keys for brand-new routes opened during this
        # trial (created lazily, only once no existing route has room).
        # Todo esto son COPIAS (`list(r["custs"])`), asi que insertar clientes
        # aca dentro no toca el estado real hasta el commit final.
        cand: dict[int, dict] = {
            ridx: {"depot": r["depot"], "custs": list(r["custs"])}
            for ridx, r in enumerate(routes) if ridx not in route_indices and r["custs"]
        }
        cand_load = {k: route_load[k] for k in cand}
        trial_depot_load = dict(depot_load)
        trial_depot_load[d] = 0  # a efectos de esta prueba, d ya no carga nada
        # Cuantas rutas (vehiculos) tiene hoy en uso cada deposito -- hace
        # falta para saber si un deposito todavia tiene un vehiculo libre
        # (V_i) antes de ofrecerle abrir una ruta nueva.
        depot_route_count: dict[int, int] = {}
        for r in routes:
            if r["custs"]:
                depot_route_count[r["depot"]] = depot_route_count.get(r["depot"], 0) + 1

        next_new_id = -1
        added_cost = 0.0
        # (cust, key): key >= 0 is an existing route index, key < 0 is a
        # synthetic id for a route created during this trial (possibly by an
        # earlier customer -- a later customer can also be matched into it
        # via the normal "existing candidates" scan below, so whether a
        # route is synthetic is determined by the sign of `key` at commit
        # time, not by which search branch first produced it).
        placements: list[tuple[int, int]] = []

        for c in sorted(displaced, key=lambda c: -inst.demands[c]):
            dem = inst.demands[c]
            # Full scan over other depots' routes, not the customer neighbour
            # lists: a customer's k nearest neighbours are almost always
            # served by the *same* depot (that's why they were assigned
            # there), so neighbour-list pruning here would abort on the first
            # displaced customer nearly every time. Depot/route counts are
            # small relative to customer counts, so scanning them directly
            # stays cheap.
            best_delta, best_key, best_pos, best_is_new = math.inf, None, -1, False

            # Opcion A: insertar a c en alguna ruta EXISTENTE de otro
            # deposito que tenga lugar (respetando Q del vehiculo y W_i del
            # deposito), en la posicion mas barata de esa ruta.
            for key, info in cand.items():
                d_new = info["depot"]
                if d_new == d or cand_load[key] + dem > Q:
                    continue
                if trial_depot_load.get(d_new, 0) + dem > depot_capacities[d_new]:
                    continue
                nodes = [d_new] + [m + cc for cc in info["custs"]] + [d_new]
                for i in range(len(nodes) - 1):
                    a, b = nodes[i], nodes[i + 1]
                    delta = edge_cost(inst, a, m + c) + edge_cost(inst, m + c, b) - edge_cost(inst, a, b)
                    if delta < best_delta:
                        best_delta, best_key, best_pos, best_is_new = delta, key, i, False

            # Opcion B: si ninguna ruta existente sirve (o de todos modos sale
            # mas barato), abrir una ruta nueva -- solo en un deposito que
            # todavia tenga un vehiculo libre (count < V_i). Costo: el fijo
            # por ruta mas el viaje de ida y vuelta desde ese deposito.
            for d_new, count in depot_route_count.items():
                if d_new == d or count >= inst.depot_max_vehicles[d_new]:
                    continue
                if trial_depot_load.get(d_new, 0) + dem > depot_capacities[d_new]:
                    continue
                delta = inst.route_fixed_cost + 2 * edge_cost(inst, d_new, m + c)
                if delta < best_delta:
                    best_delta, best_key, best_pos, best_is_new = delta, d_new, 0, True

            if best_key is None:
                return False  # este cliente no tiene donde ir -- se aborta, no se cometio nada real todavia

            if best_is_new:
                # Se registra una ruta sintetica nueva bajo un id negativo.
                # OJO: un cliente MAS ADELANTE en este mismo `displaced` puede
                # terminar cayendole a esta misma ruta a traves de la Opcion
                # A de arriba (ahora es una entrada mas de `cand`), y en ese
                # caso su placement se guarda con best_is_new=False aunque el
                # id siga siendo negativo -- por eso el commit de abajo decide
                # "es sintetica o no" mirando el SIGNO del id, no este flag.
                new_id, next_new_id = next_new_id, next_new_id - 1
                cand[new_id] = {"depot": best_key, "custs": [c]}
                cand_load[new_id] = dem
                depot_route_count[best_key] = depot_route_count.get(best_key, 0) + 1
                trial_depot_load[best_key] = trial_depot_load.get(best_key, 0) + dem
                placements.append((c, new_id))
            else:
                cand[best_key]["custs"].insert(best_pos, c)
                cand_load[best_key] += dem
                d_target = cand[best_key]["depot"]
                trial_depot_load[d_target] = trial_depot_load.get(d_target, 0) + dem
                placements.append((c, best_key))
            added_cost += best_delta
            if added_cost >= removed_cost:
                return False  # ya no compensa -- no tiene sentido seguir ubicando al resto

        if added_cost >= removed_cost - 1e-6:
            return False  # se logro ubicar a todos, pero no alcanza a compensar el cierre

        # A partir de aca se confirma todo -- ya se sabe que el cierre conviene.

        # 1) Las rutas EXISTENTES que recibieron clientes nuevos se
        #    actualizan con la version de prueba (que ya incluye tanto sus
        #    clientes originales como los reubicados).
        for ridx in cand:
            if ridx >= 0 and routes[ridx]["custs"] != cand[ridx]["custs"]:
                routes[ridx]["custs"] = cand[ridx]["custs"]
                route_load[ridx] = cand_load[ridx]

        # 2) Cada ruta sintetica (id negativo) se vuelve una ruta real,
        #    agregada al final de la lista `routes`.
        new_id_to_ridx: dict[int, int] = {}
        for key in cand:
            if key < 0:
                new_id_to_ridx[key] = len(routes)
                routes.append({"depot": cand[key]["depot"], "custs": list(cand[key]["custs"])})
                route_load.append(cand_load[key])

        # 3) route_of de cada cliente reubicado apunta ahora a su ruta real
        #    definitiva (traduciendo los ids sinteticos via new_id_to_ridx).
        for c, key in placements:
            route_of[c] = new_id_to_ridx[key] if key < 0 else key
        # 4) Las rutas de d quedan vacias (se filtran del resultado final).
        for ridx in route_indices:
            routes[ridx]["custs"] = []
            route_load[ridx] = 0
        for d_new, load in trial_depot_load.items():
            depot_load[d_new] = load
        return True

    def relocate_swap_pass() -> bool:
        """One shuffled sweep over all customers, applying the best
        improving relocate/swap move found for each. Returns whether any
        move was applied."""
        improved_any = False
        # Se recorren los clientes en orden aleatorio (no siempre el mismo)
        # para no sesgar sistematicamente que jugadas se prueban primero.
        customers = list(route_of.keys())
        random.shuffle(customers)
        for c in customers:
            if time.time() - start > time_limit:
                break
            r_old = route_of[c]
            d_old = routes[r_old]["depot"]
            dem = inst.demands[c]
            saving = removal_saving(r_old, c)

            # Poda "granular": solo se consideran como destino las rutas que
            # ya tienen a algun vecino cercano de c (segun neighbor_lists),
            # no TODAS las rutas de la instancia. Es una heuristica estandar
            # para no pagar O(clientes x rutas) en cada pasada.
            cand_routes = {route_of[c2] for c2 in neighbor_lists[c] if c2 in route_of}
            cand_routes.discard(r_old)

            best_gain = 1e-6
            best_move = None

            # RELOCATE: sacar a c de su ruta e insertarlo en otra, si el
            # ahorro de sacarlo supera el costo de insertarlo ahi.
            for r_new in cand_routes:
                d_new = routes[r_new]["depot"]
                if route_load[r_new] + dem > Q:
                    continue
                if d_new != d_old and depot_load.get(d_new, 0) + dem > depot_capacities[d_new]:
                    continue
                delta_add, pos = insertion_cost(r_new, c)
                gain = saving - delta_add
                if gain > best_gain:
                    best_gain = gain
                    best_move = ("relocate", r_new, pos)

            # SWAP: intercambiar a c con un vecino cercano c2 (pueden estar en
            # la misma ruta o en dos distintas), si el intercambio achica la
            # distancia total de ambas rutas involucradas.
            for c2 in neighbor_lists[c]:
                r2 = route_of.get(c2)
                if r2 is None or r2 == r_old:
                    continue
                d2 = routes[r2]["depot"]
                dem2 = inst.demands[c2]
                if route_load[r_old] - dem + dem2 > Q or route_load[r2] - dem2 + dem > Q:
                    continue
                if d_old != d2:
                    if (depot_load.get(d_old, 0) - dem + dem2 > depot_capacities[d_old]
                            or depot_load.get(d2, 0) - dem2 + dem > depot_capacities[d2]):
                        continue
                pos_c = routes[r_old]["custs"].index(c)
                pos_c2 = routes[r2]["custs"].index(c2)
                nodes_old = route_nodes(r_old)
                nodes_2 = route_nodes(r2) if r2 != r_old else nodes_old
                a1, b1 = nodes_old[pos_c], nodes_old[pos_c + 2]
                a2, b2 = nodes_2[pos_c2], nodes_2[pos_c2 + 2]
                before = (edge_cost(inst, a1, m + c) + edge_cost(inst, m + c, b1)
                          + edge_cost(inst, a2, m + c2) + edge_cost(inst, m + c2, b2))
                after = (edge_cost(inst, a1, m + c2) + edge_cost(inst, m + c2, b1)
                         + edge_cost(inst, a2, m + c) + edge_cost(inst, m + c, b2))
                gain = before - after
                if gain > best_gain:
                    best_gain = gain
                    best_move = ("swap", r2, c2)

            if best_move is None:
                continue  # ninguna jugada mejoro el costo para este cliente
            improved_any = True

            # Se aplica la mejor jugada encontrada, actualizando todos los
            # indices auxiliares (route_of, route_load, depot_load) para que
            # sigan reflejando el estado real.
            if best_move[0] == "relocate":
                _, r_new, pos = best_move
                d_new = routes[r_new]["depot"]
                routes[r_old]["custs"].remove(c)
                route_load[r_old] -= dem
                depot_load[d_old] -= dem
                routes[r_new]["custs"].insert(pos, c)
                route_load[r_new] += dem
                depot_load[d_new] = depot_load.get(d_new, 0) + dem
                route_of[c] = r_new
            else:
                _, r2, c2 = best_move
                d2 = routes[r2]["depot"]
                dem2 = inst.demands[c2]
                idx1 = routes[r_old]["custs"].index(c)
                idx2 = routes[r2]["custs"].index(c2)
                routes[r_old]["custs"][idx1] = c2
                routes[r2]["custs"][idx2] = c
                route_load[r_old] += dem2 - dem
                route_load[r2] += dem - dem2
                depot_load[d_old] = depot_load.get(d_old, 0) + dem2 - dem
                depot_load[d2] = depot_load.get(d2, 0) + dem - dem2
                route_of[c] = r2
                route_of[c2] = r_old
        return improved_any

    def close_depot_pass() -> bool:
        """One sweep trying to close each open depot, least-loaded first.
        Returns whether any depot was actually closed."""
        closed_any = False
        open_depots_by_load = sorted(
            {r["depot"] for r in routes if r["custs"]},
            key=lambda dd: depot_load.get(dd, 0),
        )
        for d in open_depots_by_load:
            if time.time() - start > time_limit:
                break
            if try_close_depot(d):
                closed_any = True
        return closed_any

    def current_cost() -> float:
        """Costo total del estado interno actual (mismo calculo que
        compute_total_cost, pero directo sobre `routes` sin pasar por el
        formato routes_by_depot)."""
        total = 0.0
        opened_depots = set()
        for ridx, r in enumerate(routes):
            if not r["custs"]:
                continue
            opened_depots.add(r["depot"])
            total += inst.route_fixed_cost
            nodes = route_nodes(ridx)
            total += sum(edge_cost(inst, nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))
        total += sum(inst.opening_costs[d] for d in opened_depots)
        return total

    def snapshot():
        """Copia completa del estado interno, para poder volver atras si una
        perturbacion de LNS termina siendo peor que lo que ya se tenia."""
        return (
            [{"depot": r["depot"], "custs": list(r["custs"])} for r in routes],
            dict(route_of),
            list(route_load),
            dict(depot_load),
        )

    def restore(snap) -> None:
        """Reemplaza el estado interno por una foto tomada con snapshot().
        Se muta todo en el lugar (clear + repoblar) en vez de reasignar las
        variables, porque son closures compartidas con las demas funciones
        anidadas de local_search."""
        snap_routes, snap_route_of, snap_route_load, snap_depot_load = snap
        routes.clear()
        routes.extend({"depot": r["depot"], "custs": list(r["custs"])} for r in snap_routes)
        route_of.clear()
        route_of.update(snap_route_of)
        route_load.clear()
        route_load.extend(snap_route_load)
        depot_load.clear()
        depot_load.update(snap_depot_load)

    def destroy() -> list[int]:
        """Saca de sus rutas a un grupo de clientes (10-20% del total, elegido
        al azar cada vez) para que converge_local() los tenga que reubicar
        de cero despues. Sortea entre tres formas de elegirlos, para que la
        perturbacion no sea siempre igual:
          - "random": un subconjunto al azar de toda la instancia.
          - "related": un cliente al azar mas sus vecinos geograficos
            cercanos (via neighbor_lists) -- perturba una zona concentrada.
          - "route": rutas completas al azar, hasta juntar la cantidad
            buscada -- fuerza a re-decidir la estructura de esas rutas enteras.
        No decide donde van a terminar -- eso lo hace repair() despues.
        """
        all_customers = list(route_of.keys())
        n_total = len(all_customers)
        target = max(1, min(n_total, int(n_total * random.uniform(0.08, 0.20))))
        strategy = random.choice(("random", "related", "route"))

        if strategy == "random":
            removed = random.sample(all_customers, target)
        elif strategy == "related":
            seed = random.choice(all_customers)
            removed = [seed] + [c2 for c2 in neighbor_lists[seed] if c2 in route_of][:target - 1]
        else:
            removed = []
            route_indices = [ridx for ridx, r in enumerate(routes) if r["custs"]]
            random.shuffle(route_indices)
            for ridx in route_indices:
                if len(removed) >= target:
                    break
                removed.extend(routes[ridx]["custs"])

        removed = list(dict.fromkeys(removed))  # por si alguna estrategia repite ids
        for c in removed:
            ridx = route_of.pop(c)
            routes[ridx]["custs"].remove(c)
            dem = inst.demands[c]
            route_load[ridx] -= dem
            depot_load[routes[ridx]["depot"]] -= dem
        return removed

    def repair(removed: list[int]) -> None:
        """Reinserta cada cliente sacado por destroy() en la posicion mas
        barata disponible: en una ruta existente que tenga lugar, o -- si
        ninguna sirve -- abriendo una ruta nueva (en un deposito abierto con
        vehiculo libre, o en uno cerrado, pagando su costo de apertura). Es
        la misma idea que la Opcion A/B de try_close_depot, pero aca el
        candidato puede ser CUALQUIER deposito (no solo "otro distinto de
        d"), porque no estamos cerrando nada.
        """
        depot_route_count: dict[int, int] = {}
        for r in routes:
            if r["custs"]:
                depot_route_count[r["depot"]] = depot_route_count.get(r["depot"], 0) + 1

        for c in sorted(removed, key=lambda c: -inst.demands[c]):
            dem = inst.demands[c]
            best_delta, best_ridx, best_pos = math.inf, -1, -1
            best_new_depot = -1

            for ridx, r in enumerate(routes):
                if not r["custs"] or route_load[ridx] + dem > Q:
                    continue
                d = r["depot"]
                if depot_load.get(d, 0) + dem > depot_capacities[d]:
                    continue
                nodes = route_nodes(ridx)
                for i in range(len(nodes) - 1):
                    a, b = nodes[i], nodes[i + 1]
                    delta = edge_cost(inst, a, m + c) + edge_cost(inst, m + c, b) - edge_cost(inst, a, b)
                    if delta < best_delta:
                        best_delta, best_ridx, best_pos = delta, ridx, i

            for d in range(inst.m):
                count = depot_route_count.get(d, 0)
                if count >= inst.depot_max_vehicles[d] or depot_load.get(d, 0) + dem > depot_capacities[d]:
                    continue
                extra = inst.opening_costs[d] if count == 0 else 0.0
                delta = inst.route_fixed_cost + 2 * edge_cost(inst, d, m + c) + extra
                if delta < best_delta:
                    best_delta, best_ridx, best_new_depot = delta, -1, d

            if best_ridx == -1 and best_new_depot == -1:
                # No deberia pasar (la construccion inicial ya garantiza que
                # la demanda total entra en algun lado), pero por las dudas
                # se fuerza la insercion en la ruta con menos carga (puede
                # violar Q/W_i) en vez de abortar el proceso.
                best_ridx = min(range(len(routes)), key=lambda i: route_load[i])
                best_pos = 0

            if best_new_depot != -1:
                new_ridx = len(routes)
                routes.append({"depot": best_new_depot, "custs": [c]})
                route_load.append(dem)
                depot_load[best_new_depot] = depot_load.get(best_new_depot, 0) + dem
                depot_route_count[best_new_depot] = depot_route_count.get(best_new_depot, 0) + 1
                route_of[c] = new_ridx
            else:
                r = routes[best_ridx]
                r["custs"].insert(best_pos, c)
                route_load[best_ridx] += dem
                d = r["depot"]
                depot_load[d] = depot_load.get(d, 0) + dem
                route_of[c] = best_ridx

    def converge_local() -> None:
        """Alterna relocate/swap y cierre de depositos hasta que ninguno de
        los dos encuentre mas mejoras (o se acabe el tiempo)."""
        while time.time() - start < time_limit:
            while relocate_swap_pass() and time.time() - start < time_limit:
                pass  # sigue pasando mientras encuentre alguna mejora
            if time.time() - start >= time_limit or not close_depot_pass():
                break  # nada mas para mejorar (o se acabo el tiempo)
            # Si se cerro algun deposito, sus clientes se movieron en bloque a
            # otro lado -- se vuelve a converger relocate/swap sobre el nuevo
            # arreglo antes de intentar cerrar otro deposito.

    # Fase 1: relocate/swap/cierre hasta el primer optimo local.
    # Fase 2 (LNS): si sobra tiempo, se perturba esa solucion a proposito
    # (destroy) y se vuelve a optimizar (repair + converge_local), quedandose
    # siempre con la mejor version vista hasta ahora. Esto es lo que permite
    # escapar de optimos locales a los que relocate/swap/close solos no
    # pueden salir (ninguna de esas jugadas, por si sola, empeora el costo
    # nunca, asi que sin la perturbacion de LNS quedarian encerradas ahi).
    start = time.time()
    converge_local()

    best_cost = current_cost()
    best_snapshot = snapshot()
    while time.time() - start < time_limit:
        removed = destroy()
        repair(removed)
        converge_local()
        cost = current_cost()
        if cost < best_cost - 1e-6:
            best_cost = cost
            best_snapshot = snapshot()
        else:
            restore(best_snapshot)  # la perturbacion no ayudo -- se vuelve al mejor conocido
    restore(best_snapshot)

    # Se arma el resultado final descartando rutas vacias (depositos
    # cerrados, o rutas que quedaron sin clientes tras un relocate).
    result: dict[int, list[list[int]]] = {}
    for r in routes:
        if r["custs"]:
            result.setdefault(r["depot"], []).append(r["custs"])
    # Se vuelve a correr nearest-neighbour + 2-opt en cada ruta: el orden de
    # visita interno que se arrastro durante relocate/swap/close es solo un
    # subproducto de como se insertaron los clientes, no una secuencia
    # optimizada -- por eso se recalcula desde cero aca.
    return {d: [order_route(inst, d, r) for r in rlist] for d, rlist in result.items()}


# ---------------------------------------------------------------------------
# Step 3: assemble + write .sol
# ---------------------------------------------------------------------------

def build_solution(inst, seed: int = 0, time_limit: float = 20.0, improve: bool = True):
    """Orquesta el pipeline completo: construccion golosa (facility location
    + reparacion de bin packing + rutas) y despues, si hay tiempo, la mejora
    con local_search (relocate/swap/cierre de deposito)."""
    random.seed(seed)
    assignment = solve_facility_location(inst)
    assignment = _repair_bin_packing(inst, assignment)

    routes_by_depot: dict[int, list[list[int]]] = {}
    for depot, customers in assignment.items():
        bins = pack_routes(inst, depot, customers)  # strict=False: last-resort fallback
        ordered = [order_route(inst, depot, b) for b in bins]
        routes_by_depot[depot] = [r for r in ordered if r]

    if improve and time_limit > 0:
        neighbor_lists = build_neighbor_lists(inst)
        routes_by_depot = local_search(inst, routes_by_depot, neighbor_lists, time_limit=time_limit)
    return routes_by_depot


def _repair_bin_packing(inst, assignment: dict[int, list[int]]) -> dict[int, list[int]]:
    """Fix depots whose customer set can't be packed into V_i routes of
    capacity Q, even though total demand fits (bin packing is NP-hard, so the
    facility-location step's capacity check is necessary but not sufficient).

    Relocates the customer causing the overflow to the nearest other depot
    with spare effective capacity, opening a new depot if none has room.
    """
    m = inst.m
    eff_capacity = [min(inst.depot_capacities[d], inst.depot_max_vehicles[d] * inst.vehicle_capacity)
                    for d in range(m)]
    assignment = {d: list(cs) for d, cs in assignment.items()}
    used = {d: sum(inst.demands[c] for c in cs) for d, cs in assignment.items()}
    open_depots = set(assignment.keys())

    pending = [d for d, cs in assignment.items() if cs]
    seen_rounds = 0
    max_rounds = 4 * sum(len(cs) for cs in assignment.values()) + 50

    while pending and seen_rounds < max_rounds:
        seen_rounds += 1
        d = pending.pop(0)
        customers = assignment.get(d, [])
        if not customers:
            continue
        if pack_routes(inst, d, customers, strict=True) is not None:
            continue  # already feasible

        # Relocate the largest-demand customer first (fastest to relieve load).
        customers_sorted = sorted(customers, key=lambda c: -inst.demands[c])
        moved = False
        for c in customers_sorted:
            dem = inst.demands[c]
            # Primero busca un deposito YA abierto con lugar; si ninguno
            # sirve, recien ahi abre uno nuevo (el mas cercano a c que tenga
            # capacidad suficiente).
            others = sorted(
                (dd for dd in open_depots if dd != d),
                key=lambda dd: edge_cost(inst, dd, m + c),
            )
            target = next((dd for dd in others if used.get(dd, 0) + dem <= eff_capacity[dd]), None)
            if target is None:
                closed = sorted(
                    (dd for dd in range(m) if dd not in open_depots),
                    key=lambda dd: edge_cost(inst, dd, m + c),
                )
                target = next((dd for dd in closed if eff_capacity[dd] >= dem), None)
                if target is not None:
                    open_depots.add(target)
                    assignment[target] = []
                    used[target] = 0

            if target is not None:
                assignment[d].remove(c)
                used[d] -= dem
                assignment[target].append(c)
                used[target] = used.get(target, 0) + dem
                pending.append(target)
                pending.append(d)
                moved = True
                break

        if not moved:
            break  # nothing we can do; pack_routes' strict=False fallback will handle it

    return {d: cs for d, cs in assignment.items() if cs}


def compute_total_cost(inst, routes_by_depot: dict[int, list[list[int]]]) -> float:
    """Costo total = apertura de cada deposito usado + costo fijo por cada
    ruta + distancia recorrida por cada ruta (deposito -> clientes -> deposito)."""
    total = 0.0
    for depot, routes in routes_by_depot.items():
        if routes:
            total += inst.opening_costs[depot]
        for r in routes:
            total += inst.route_fixed_cost
            nodes = [depot] + [inst.m + c for c in r] + [depot]
            for i in range(len(nodes) - 1):
                total += edge_cost(inst, nodes[i], nodes[i + 1])
    return total


def write_solution(inst, routes_by_depot: dict[int, list[list[int]]], out_path: Path):
    """Escribe el archivo .sol en el formato oficial (ver especificacion_tecnica.md)."""
    total_routes = sum(len(r) for r in routes_by_depot.values())
    depots_opened = sum(1 for r in routes_by_depot.values() if r)
    cost = compute_total_cost(inst, routes_by_depot)

    lines = [f"# instance={inst.name}", f"COST : {cost:.4f}",
             f"DEPOTS_OPENED : {depots_opened}", f"ROUTES : {total_routes}"]
    for depot in sorted(routes_by_depot):
        routes = routes_by_depot[depot]
        if not routes:
            continue
        lines.append(f"DEPOT {depot + 1}")
        for r in routes:
            cust_ids = " ".join(str(inst.m + c + 1) for c in r)
            lines.append(f"ROUTE : {cust_ids}")
    lines.append("EOF")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cost


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--time-limit", type=float, default=20.0,
                     help="seconds of local-search improvement after construction (default 20, 0 to disable)")
    args = ap.parse_args()

    t0 = time.time()
    inst = parse_instance(args.instance)
    routes_by_depot = build_solution(inst, time_limit=args.time_limit, improve=args.time_limit > 0)
    cost = write_solution(inst, routes_by_depot, args.output)
    elapsed = time.time() - t0
    print(f"{inst.name}: cost={cost:.2f} depots={sum(1 for r in routes_by_depot.values() if r)} "
          f"routes={sum(len(r) for r in routes_by_depot.values())} time={elapsed:.2f}s")

    if args.verify:
        result = verify_solution(inst, args.output)
        print(f"feasible={result.feasible} recomputed_cost={result.recomputed_cost:.2f}")
        for e in result.errors:
            print("  ERROR:", e)


if __name__ == "__main__":
    main()
