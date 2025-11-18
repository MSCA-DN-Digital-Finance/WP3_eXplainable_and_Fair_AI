from tsg.generators import BaseGenerator
from typing import Sequence, Any, Callable, Union

# Define custom generator to feed twin trajectories with same noise

class NoiseTrajectoryWrapper(BaseGenerator):

    def __init__(self, generator, noise_traj):
        self.generator = generator
        self.noise_traj = noise_traj
        self.t = 0

    def generate_value(self, last_value):
        base_value = self.generator.generate_value(last_value)
        noise = self.noise_traj[self.t]
        self.t += 1
        return base_value + noise

    def reset(self):
        self.generator.reset()
        self.t = 0



# gen_factory can be:
#   - Callable[[float], Any]                       (single generator)
#   - Sequence[Callable[[float], Any]]             (multi-generator)
GenFactory = Union[Callable[[float], Any], Sequence[Callable[[float], Any]]]