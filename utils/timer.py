import time
from utils.logger import get_logger

logger = get_logger("perf")

class Timer:

    def __init__(self, prefix=""):
        self.last = time.perf_counter()
        self.prefix = prefix

    def log(self, name):

        now = time.perf_counter()
        elapsed = (now - self.last) * 1000

        logger.debug("%s %s: %.2f ms", self.prefix, name, elapsed)

        self.last = now