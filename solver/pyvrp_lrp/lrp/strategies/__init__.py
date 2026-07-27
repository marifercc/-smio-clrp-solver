from typing import Protocol

import pyvrp

from lrp.strategies.decomposition import decomposition
from lrp.strategies.direct import direct
from lrp.strategies.multi_start import multi_start


class Strategy(Protocol):
    """
    Interface that every strategy must implement. A strategy turns an instance
    into a solution, using at most ``max_runtime`` seconds.
    """

    def __call__(
        self,
        data: pyvrp.ProblemData,
        max_runtime: float,
        seed: int = 0,
    ) -> pyvrp.Result: ...


# Maps a strategy name to its implementation. Add new strategies here.
STRATEGIES: dict[str, Strategy] = {
    "decomposition": decomposition,
    "direct": direct,
    "multi_start": multi_start,
}
