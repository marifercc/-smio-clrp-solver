import pyvrp
import pyvrp.stop


def direct(
    data: pyvrp.ProblemData, max_runtime: float, seed: int = 0
) -> pyvrp.Result:
    """
    Solves the LRP instance directly with PyVRP.
    """
    stop = pyvrp.stop.MaxRuntime(max_runtime)
    return pyvrp.solve(data, stop=stop, seed=seed)
