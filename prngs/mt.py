class MersenneTwister:
    """Thin wrapper around Python's built-in random for uniform interface."""
    import random as _random
    def __init__(self, seed):
        self._rng = __import__('random').Random(seed)
    
    def random(self):
        return self._rng.random()
