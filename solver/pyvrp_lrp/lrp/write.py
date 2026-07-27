from pathlib import Path

import pyvrp


def write(
    where: Path,
    data: pyvrp.ProblemData,
    result: pyvrp.Result,
    scale: int = 100,
):
    """
    Writes a PyVRP result in the SMIO challenge solution format.

    PyVRP costs are divided by ``scale`` to recover the challenge units;
    see ``read.py`` for the details.
    """

    routes = result.best.routes()
    depots = sorted({route.start_depot() for route in routes})

    lines = [
        f"# instance={where.stem}",
        f"COST : {result.cost() / scale:.2f}",
        f"DEPOTS_OPENED : {len(depots)}",
        f"ROUTES : {len(routes)}",
    ]

    for depot in depots:
        # A location's name holds its original instance index. We use names
        # rather than PyVRP's internal indices so the output stays correct even
        # when we solve only a subset of the problem.
        idx = data.location(depot).name
        lines.append(f"DEPOT {idx}")

        for route in routes:
            if route.start_depot() == depot:
                idcs = [data.location(loc).name for loc in route.visits()]
                lines.append(f"  ROUTE : {' '.join(idcs)}")

    lines.append("EOF")
    where.write_text("\n".join(lines) + "\n")
