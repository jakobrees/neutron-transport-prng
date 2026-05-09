class MiddleSquare:
    """
    Von Neumann's middle-square method, 8 decimal digits.
    Historically authentic: first ENIAC run spring 1948.
    
    Reference: Hayes, B. "The Middle of the Square", bit-player.org, 2022.
    """
    def __init__(self, seed):
        assert 10_000_000 <= seed <= 99_999_999, "Need 8-digit seed"
        self.state = seed
        self.width = 8          # decimal digits, must be even
        self.modulus = 10 ** (3 * self.width // 2)   # 10^12
        self.divisor = 10 ** (self.width // 2)        # 10^4
        self._initial = seed
        self._count = 0

    def random(self):
        """Returns float in [0, 1) and advances state."""
        squared = self.state ** 2
        self.state = (squared % self.modulus) // self.divisor
        self._count += 1
        return self.state / 1e8   # normalize by 10^8

    def detect_cycle(self, max_iter=500_000):
        """Floyd's tortoise and hare cycle detection."""
        seen = {}
        s = self._initial
        for i in range(max_iter):
            if s in seen:
                return i - seen[s]   # cycle length
            seen[s] = i
            squared = s ** 2
            s = (squared % self.modulus) // self.divisor
        return None