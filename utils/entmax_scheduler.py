import math


def get_entmax_weight_scheduler(sheduler):
    tokens = sheduler.split("_")

    if tokens[0] == "linear":
        return LinearScheduler(float(tokens[1]), float(tokens[2]), int(tokens[3]))
    elif tokens[0] == "cosine":
        return CosineScheduler(float(tokens[1]), float(tokens[2]), int(tokens[3]))
    elif tokens[0] == "linearcosine":
        return LinearCosineScheduler(
            float(tokens[1]),
            float(tokens[2]),
            int(tokens[3]),
            float(tokens[4]),
            int(tokens[5]),
        )
    else:
        raise ValueError(f"Unknown sheduler {sheduler}")


class LinearScheduler:
    def __init__(self, start_val, end_val, n_steps):
        # entmax unstable when alpha == 1, so we add 1e-8
        self.start_val = start_val + 1e-8
        self.end_val = end_val
        self.n_steps = n_steps

    def __call__(self, step):
        if step >= self.n_steps:
            return self.end_val
        else:
            return (
                self.start_val + step * (self.end_val - self.start_val) / self.n_steps
            )


class CosineScheduler:
    def __init__(self, start_val, end_val, n_steps):
        self.start_val = start_val + 1e-8
        self.end_val = end_val
        self.n_steps = n_steps

    def __call__(self, step):
        if step >= self.n_steps:
            return self.end_val
        else:
            return self.end_val + 0.5 * (self.start_val - self.end_val) * (
                1 + math.cos(math.pi * step / self.n_steps)
            )


class LinearCosineScheduler:
    def __init__(self, start_val, middle_val, n_steps, end_val, n_steps2):
        self.start_val = start_val + 1e-8
        self.middle_val = middle_val
        self.end_val = end_val
        self.n_steps = n_steps
        self.n_steps2 = n_steps2

    def __call__(self, step):
        if step >= self.n_steps + self.n_steps2:
            return self.end_val
        elif step >= self.n_steps:
            return self.end_val + 0.5 * (self.middle_val - self.end_val) * (
                1 + math.cos(math.pi * (step - self.n_steps) / self.n_steps2)
            )
        else:
            return (
                self.start_val
                + step * (self.middle_val - self.start_val) / self.n_steps
            )
