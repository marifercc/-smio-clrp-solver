from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyvrp


def read(where: Path, scale: int = 100) -> pyvrp.ProblemData:
    """
    Reads an SMIO CLRP instance into a PyVRP ``ProblemData`` object.

    Costs and distances are multiplied by ``scale`` because PyVRP uses integer
    costs internally. The default scale of 100 preserves the SMIO file format's
    two-decimal fixed costs and one-decimal distances exactly.
    """

    instance = _parse_instance(where)
    clients = [
        pyvrp.Client(row.x, row.y, delivery=[row.demand], name=str(row.idx))
        for row in instance.customers
    ]
    depots = [
        pyvrp.Depot(
            row.x,
            row.y,
            fixed_cost=int(row.fixed_cost * scale),
            capacity=[row.capacity],
            name=str(row.idx),
        )
        for row in instance.depots
    ]
    vehicle_types = [
        pyvrp.VehicleType(
            num_available=row.max_vehicles,
            capacity=[instance.vehicle_capacity],
            start_depot=depot,
            end_depot=depot,
            fixed_cost=int(instance.route_fixed_cost * scale),
            name=str(row.idx),
        )
        for depot, row in enumerate(instance.depots)
    ]

    distances = np.rint(instance.distance_matrix() * scale).astype(np.int64)

    return pyvrp.ProblemData(
        clients,
        depots,
        vehicle_types,
        [distances],
        [np.zeros_like(distances)],
    )


@dataclass
class _Depot:
    idx: int
    x: float
    y: float
    fixed_cost: Decimal
    capacity: int
    max_vehicles: int


@dataclass
class _Customer:
    idx: int
    x: float
    y: float
    demand: int


@dataclass
class _Instance:
    num_depots: int
    num_customers: int
    vehicle_capacity: int
    route_fixed_cost: Decimal
    distance_format: str
    depots: list[_Depot]
    customers: list[_Customer]
    matrix_lines: list[str]

    def distance_matrix(self) -> np.ndarray:
        if self.distance_format == "FULL_MATRIX":
            size = self.num_depots + self.num_customers
            values = np.fromstring(" ".join(self.matrix_lines), sep=" ")

            if values.size != size * size:
                raise ValueError("DISTANCE_SECTION has the wrong size.")

            return values.reshape(size, size)

        coords = np.array(
            [(row.x, row.y) for row in self.depots]
            + [(row.x, row.y) for row in self.customers],
            dtype=np.float64,
        )

        # Pairwise Euclidean.
        sq_sum = (coords**2).sum(axis=1)
        sq_dist = np.add.outer(sq_sum, sq_sum) - 2 * (coords @ coords.T)
        np.fill_diagonal(sq_dist, 0)

        return np.round(np.sqrt(sq_dist), 1)


def _parse_instance(where: Path) -> _Instance:
    fields = {}
    depots = []
    customers = []
    matrix_lines = []
    section = "header"

    for line in _content_lines(where):
        if line == "EOF":
            break

        if line in {"DEPOT_SECTION", "CUSTOMER_SECTION", "DISTANCE_SECTION"}:
            section = line
            continue

        if section == "header":
            key, value = line.split(":", 1)
            fields[key.strip().upper()] = value.strip()
        elif section == "DEPOT_SECTION":
            idx, x, y, fixed_cost, capacity, max_vehicles = line.split()
            depots.append(
                _Depot(
                    int(idx),
                    float(x),
                    float(y),
                    Decimal(fixed_cost),
                    int(capacity),
                    int(max_vehicles),
                )
            )
        elif section == "CUSTOMER_SECTION":
            idx, x, y, demand = line.split()
            customers.append(
                _Customer(int(idx), float(x), float(y), int(demand))
            )
        elif section == "DISTANCE_SECTION":
            matrix_lines.append(line)

    distance_format = fields["DISTANCE_FORMAT"].upper()
    if distance_format not in {"COORDS", "FULL_MATRIX"}:
        raise ValueError("DISTANCE_FORMAT must be COORDS or FULL_MATRIX.")

    return _Instance(
        num_depots=int(fields["DEPOTS"]),
        num_customers=int(fields["CUSTOMERS"]),
        vehicle_capacity=int(fields["VEHICLE_CAPACITY"]),
        route_fixed_cost=Decimal(fields["ROUTE_FIXED_COST"]),
        distance_format=distance_format,
        depots=depots,
        customers=customers,
        matrix_lines=matrix_lines,
    )


def _content_lines(where: Path):
    with where.open() as fh:
        for raw in fh:
            line = raw.strip()

            if line and not line.startswith("#"):
                yield line
