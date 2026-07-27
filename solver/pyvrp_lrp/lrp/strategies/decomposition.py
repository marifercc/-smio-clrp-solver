import numpy as np
import pyvrp
import pyvrp.stop


def decomposition(
    data: pyvrp.ProblemData, max_runtime: float, seed: int = 0
) -> pyvrp.Result:
    """
    Selects a promising depot subset, then solves the resulting VRP.
    """
    depot_idcs = _select_depots(data)

    # In all LRP instances, each depot has exactly one vehicle type starting
    # and ending there. Filtering vehicle types is therefore enough to disable
    # certain depots, while keeping all other data unchanged.
    vehicle_types = [data.vehicle_type(idx) for idx in depot_idcs]
    reduced_data = data.replace(vehicle_types=vehicle_types)
    stop = pyvrp.stop.MaxRuntime(max_runtime)
    return pyvrp.solve(reduced_data, stop=stop, seed=seed)


def _select_depots(data: pyvrp.ProblemData) -> list[int]:
    """
    Keeps every depot that is the nearest depot for at least one customer.

    More depots are added by fixed cost as needed until the selected subset has
    enough depot and vehicle capacity to serve all demand.
    """
    distances = data.distance_matrix(0)
    depot2client = distances[: data.num_depots, data.num_depots :]
    nearest_depots = np.argmin(depot2client, axis=0)
    selected = sorted({int(depot) for depot in nearest_depots})
    remaining = sorted(
        set(range(data.num_depots)) - set(selected),
        key=lambda depot: data.depots()[depot].fixed_cost,
    )

    for depot in remaining:
        if _has_enough_capacity(data, selected):
            break

        selected.append(depot)

    selected.sort()
    return selected


def _has_enough_capacity(
    data: pyvrp.ProblemData, depot_idcs: list[int]
) -> bool:
    for dim in range(data.num_load_dimensions):
        demand = sum(client.delivery[dim] for client in data.clients())
        capacity = 0

        for depot in depot_idcs:
            depot_capacity = data.depots()[depot].capacity[dim]
            veh_type = data.vehicle_types()[depot]
            veh_capacity = veh_type.num_available * veh_type.capacity[dim]
            capacity += min(depot_capacity, veh_capacity)

        if capacity < demand:
            return False

    return True
