import random

from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs, legal):
        return self.rng.choice(legal)
