class LCG:
    """Linear Congruential Generator. Hull-Dobell conditions for full period:
       1. c and m are coprime
       2. a-1 divisible by all prime factors of m
       3. if m divisible by 4, a-1 divisible by 4
    """
    def __init__(self, seed, a=1664525, c=1013904223, m=2**32):
        # Numerical Recipes parameters — satisfy Hull-Dobell
        self.state = seed
        self.a = a
        self.c = c
        self.m = m
    
    def random(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m
