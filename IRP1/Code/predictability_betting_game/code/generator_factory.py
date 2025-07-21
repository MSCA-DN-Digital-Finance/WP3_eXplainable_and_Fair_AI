from tsg.generators import (
    LinearTrendGenerator,
    OrnsteinUhlenbeckGenerator,
    GaussianNoiseGenerator,
    SinusoidalGenerator,
)
from tsg.meta_generators import NoisyGenerator, MarkovSwitchGenerator

GENERATOR_MAP = {
    "Linear Trend Generator": LinearTrendGenerator,
    "Ornstein-Uhlenbeck Generator": OrnsteinUhlenbeckGenerator,
    "Gaussian Noise Generator": GaussianNoiseGenerator,
    "Sinusoidal Generator": SinusoidalGenerator,
}


class NoisyWrapperGenerator(NoisyGenerator):
    def __init__(self, base_class, params, start_value=0.0):
        noise_mu = params.get("noise_mu", 0.0)
        noise_sigma = params.get("noise_sigma", 1.0)

        base_params = params.get("params", {}).copy()
        base_gen = base_class(**base_params)
        super().__init__(base_gen, noise_mu=noise_mu, noise_sigma=noise_sigma)


def create_generator(generator_name, params):
    if generator_name == "Noisy Markov Regime-Switching Generator":
        regimes = params["regimes"]
        generator_classes = []
        generator_params_list = []

        for regime in regimes:
            class_name = regime["generator"]
            regime_params = regime["params"]
            generator_classes.append(GENERATOR_MAP[class_name])
            generator_params_list.append(regime_params)

        return MarkovSwitchGenerator(
            generator_classes,
            generator_params_list,
            params["transition_matrix"]
        )

    elif generator_name == "Noisy Generator":
        base_class_name = params["generator"]
        base_class = GENERATOR_MAP[base_class_name]
        return NoisyWrapperGenerator(base_class, params, start_value=params.get("start_value", 0.0))

    elif generator_name in GENERATOR_MAP:
        gen_class = GENERATOR_MAP[generator_name]
        return gen_class(**params)

    else:
        raise ValueError(f"Unknown generator type: {generator_name}")
