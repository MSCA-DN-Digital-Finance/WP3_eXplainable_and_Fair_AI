from tsg.generators import (
    LinearTrendGenerator,
    ConstantGenerator,
    PeriodicTrendGenerator,
    OrnsteinUhlenbeckGenerator,
    RandomWalkGenerator,
)

def create_generator(generator_name, params=None):
    if params is None:
        params = {}

    if generator_name == "Linear Trend Generator":
        return LinearTrendGenerator(**params)
    elif generator_name == "Constant Generator":
        return ConstantGenerator()
    elif generator_name == "Periodic Trend Generator":
        return PeriodicTrendGenerator(**params)
    elif generator_name == "Ornstein-Uhlenbeck Generator":
        return OrnsteinUhlenbeckGenerator(**params)
    elif generator_name == "Random Walk Generator":
        return RandomWalkGenerator(**params)
    else:
        raise ValueError(f"Unknown generator type: {generator_name}")


