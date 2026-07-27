import numpy as np
import pyvrp
import pyvrp.stop


def multi_start(
    data: pyvrp.ProblemData, max_runtime: float, seed: int = 0
) -> pyvrp.Result:
    """
    Evaluates multiple random depot subsets quickly,
    then deeply optimizes the best one.
    """
    rng = np.random.default_rng(seed)

    num_samples = 5
    skimming_time_per_sample = min(2.0, (max_runtime * 0.3) / num_samples)
    remaining_time = max_runtime - (num_samples * skimming_time_per_sample)

    best_cost = float("inf")
    best_subset = None
    best_reduced_data = None

    # Generate some candidates. Include the default greedy one too.
    candidates = []

    # 1. Add greedy (default decomposition)
    greedy_subset = _select_depots_greedy(data)
    candidates.append(greedy_subset)

    # 2. Add randomized candidates
    for _ in range(num_samples - 1):
        candidates.append(_select_depots_randomized(data, rng))

    # Evaluate each subset with a short run
    for depot_idcs in candidates:
        vehicle_types = [data.vehicle_type(idx) for idx in depot_idcs]
        reduced_data = data.replace(vehicle_types=vehicle_types)

        stop = pyvrp.stop.MaxRuntime(skimming_time_per_sample)
        # Suppress warnings if possible, but let's just run it
        result = pyvrp.solve(reduced_data, stop=stop, seed=seed)

        if result.is_feasible() and result.cost() < best_cost:
            best_cost = result.cost()
            best_subset = depot_idcs
            best_reduced_data = reduced_data

    # If none were feasible, fallback to greedy
    if best_reduced_data is None:
        best_subset = candidates[0]
        vehicle_types = [data.vehicle_type(idx) for idx in best_subset]
        best_reduced_data = data.replace(vehicle_types=vehicle_types)

    # Deep optimization on the best subset
    stop = pyvrp.stop.MaxRuntime(remaining_time)
    return pyvrp.solve(best_reduced_data, stop=stop, seed=seed)


def _select_depots_greedy(data: pyvrp.ProblemData) -> list[int]:
    """
    Same as the original decomposition selection.
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


def _select_depots_randomized(
    data: pyvrp.ProblemData, rng: np.random.Generator
) -> list[int]:
    """
    Selects a randomized promising depot subset.
    Like greedy, but with perturbations to nearest depots and cost ordering.
    """
    distances = data.distance_matrix(0)
    depot2client = distances[: data.num_depots, data.num_depots :]

    # Instead of ALWAYS taking the nearest, sometimes take the 2nd or 3rd
    selected_set = set()
    for client_idx in range(depot2client.shape[1]):
        client_distances = depot2client[:, client_idx]
        # Get top 3 nearest depots
        top_k = np.argsort(client_distances)[:3]

        # Pick one of the top-k randomly, favoring the closest
        probabilities = [0.7, 0.2, 0.1]
        # Adjust probabilities if k < 3
        if len(top_k) < 3:
            probabilities = probabilities[: len(top_k)]
            probabilities = [p / sum(probabilities) for p in probabilities]

        chosen_depot = rng.choice(top_k, p=probabilities)
        selected_set.add(int(chosen_depot))

    selected = sorted(selected_set)

    # For remaining depots, add noise to fixed costs before sorting
    remaining = list(set(range(data.num_depots)) - set(selected))

    def noisy_cost(depot_idx):
        cost = data.depots()[depot_idx].fixed_cost
        # Add uniform noise between 0% and 50% of the cost
        noise_factor = rng.uniform(1.0, 1.5)
        return cost * noise_factor

    remaining.sort(key=noisy_cost)

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
