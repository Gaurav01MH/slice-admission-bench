"""Baseline admission policies under bandwidth and compute constraints."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Budget:
    bandwidth: int
    compute: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.bandwidth, self.compute)):
            raise ValueError("bandwidth and compute budgets must be positive integers")


@dataclass(frozen=True)
class Request:
    bandwidth: int
    compute: int
    reward: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.bandwidth, self.compute, self.reward)):
            raise ValueError("request bandwidth, compute, and reward must be positive integers")


def select_fifo(batch: Iterable[Request], budget: Budget) -> tuple[Request, ...]:
    """Admit each request that fits the remaining capacity in arrival order."""
    selected = []
    bandwidth, compute = budget.bandwidth, budget.compute
    for request in batch:
        if request.bandwidth <= bandwidth and request.compute <= compute:
            selected.append(request)
            bandwidth -= request.bandwidth
            compute -= request.compute
    return tuple(selected)


def select_density(batch: Iterable[Request], budget: Budget) -> tuple[Request, ...]:
    """Sort a known batch by reward per normalized resource unit, then admit."""
    def density(request: Request) -> Fraction:
        normalized_cost = request.bandwidth * budget.compute + request.compute * budget.bandwidth
        return Fraction(request.reward, normalized_cost)

    return select_fifo(sorted(batch, key=density, reverse=True), budget)


def select_optimal(batch: Sequence[Request], budget: Budget) -> tuple[Request, ...]:
    """Find the highest feasible offline reward for integer demands and budgets."""
    states: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {(0, 0): (0, ())}
    for index, request in enumerate(batch):
        # Freeze the previous states so this request can be selected only once.
        for (bandwidth, compute), (reward, selection) in tuple(states.items()):
            used = (bandwidth + request.bandwidth, compute + request.compute)
            if used[0] > budget.bandwidth or used[1] > budget.compute:
                continue
            new_reward = reward + request.reward
            current = states.get(used)
            if current is None or new_reward > current[0]:
                states[used] = (new_reward, selection + (index,))

    _, best = max(states.values(), key=lambda state: state[0])
    return tuple(batch[index] for index in best)
