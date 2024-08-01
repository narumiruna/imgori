import time

import torch
from loguru import logger


class Average:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        self.sum += value
        self.count += 1

    def compute(self) -> float:
        if self.count == 0:
            return float("inf")
        return self.sum / self.count


def sync_perf_counter() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.perf_counter()


def timeit(func):
    average = Average()

    def wrapper(*args, **kwargs):
        start = sync_perf_counter()
        output = func(*args, **kwargs)
        t = sync_perf_counter() - start

        average.update(t)

        logger.debug(
            "{} took {:.6f} seconds, average: {:.6f} seconds.",
            func.__qualname__,
            t,
            average.compute(),
        )
        return output

    return wrapper
