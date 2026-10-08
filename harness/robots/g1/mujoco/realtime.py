"""Pace interactive physics against monotonic wall time without catch-up bursts."""
import math
import time


class RealtimePacer:
    def __init__(self, simulation_time, *, clock=time.monotonic, sleep=time.sleep):
        self.clock = clock
        self.sleep = sleep
        self.last_simulation = float(simulation_time)
        self.last_wall = clock()

    def sync(self, simulation_time):
        now = self.clock()
        simulation_time = float(simulation_time)
        elapsed = simulation_time - self.last_simulation
        if not math.isfinite(elapsed):
            raise ValueError("non-finite simulation clock")
        # Reset starts a new epoch. Slow frames never accumulate catch-up credit.
        if elapsed > 0:
            delay = elapsed - (now - self.last_wall)
            if delay > 0:
                self.sleep(delay)
        self.last_wall = self.clock()
        self.last_simulation = simulation_time
