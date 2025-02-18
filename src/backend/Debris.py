from src.backend.common.Pose import Pose
from src.backend.common.Distributions import NormalDistribution

# Debris
class Debris:
    def __init__(self, debris_id: int, pose: Pose):
        self.id = debris_id
        self.pose = pose
        
# Debris Morphology and its subclasses
class DebrisMorphology:
    def __init__(self, diameter: NormalDistribution, color: Color):
        self.diameter = diameter
        self.color = color

    def get_sample(self):
        return {
            "diameter": self.diameter.random_samples(1)[0],
            "color": self.color,
        }


class SmallDebris(DebrisMorphology):
    pass


class MediumDebris(DebrisMorphology):
    pass


class BigDebris(DebrisMorphology):
    pass