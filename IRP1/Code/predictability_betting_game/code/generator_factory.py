from tsg.generators import (
    LinearTrendGenerator,
    ConstantGenerator,
    PeriodicTrendGenerator,
    OrnsteinUhlenbeckGenerator,
    RandomWalkGenerator
)

from tsg.meta_generators import MarkovSwitchGenerator
from tsg.modifiers import GaussianNoise


GENERATOR_CLASS_MAP = {
    "Linear Trend Generator": LinearTrendGenerator,
    "Constant Generator": ConstantGenerator,
    "Periodic Trend Generator": PeriodicTrendGenerator,
    "Ornstein-Uhlenbeck Generator": OrnsteinUhlenbeckGenerator,
    "Random Walk Generator": RandomWalkGenerator,
}


def create_generator(generator_name, params=None):
    if params is None:
        params = {}

    if generator_name in GENERATOR_CLASS_MAP:
        return GENERATOR_CLASS_MAP[generator_name](**params)

    elif generator_name == "Markov Regime-Switching Generator":
        return _create_markov_switch_generator(params)

    elif generator_name == "Noisy Markov Regime-Switching Generator":
        markov_gen = _create_markov_switch_generator(params)
        return _wrap_with_noise(markov_gen, params)

    else:
        raise ValueError(f"Unknown generator type: {generator_name}")


def _create_markov_switch_generator(params):
    regimes = params.get("regimes", [])
    transition_matrix = params.get("transition_matrix")

    if not regimes or transition_matrix is None:
        raise ValueError("MarkovSwitchGenerator requires 'regimes' and 'transition_matrix'.")

    generator_classes = []
    generator_params = []

    for regime in regimes:
        gen_name = regime["generator"]
        gen_params = regime.get("params", {})

        if gen_name not in GENERATOR_CLASS_MAP:
            raise ValueError(f"Unsupported regime generator: {gen_name}")

        generator_classes.append(GENERATOR_CLASS_MAP[gen_name])
        generator_params.append(gen_params)

    return MarkovSwitchGenerator(generator_classes, generator_params, transition_matrix=transition_matrix)


def _wrap_with_noise(generator, params):
    noise_mu = params.get("noise_mu", 0.0)
    noise_sigma = params.get("noise_sigma", 0.1)
    return GaussianNoise(generator, mu=noise_mu, sigma=noise_sigma)
