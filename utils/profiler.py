import time
from utils.logger import get_logger

logger = get_logger("perf")


class InferenceProfiler:

    def __init__(self):

        self.request_start = None
        self.prefill_end = None

        self.decode_start = None
        self.decode_end = None

        self.token_count = 0

    def start_request(self):

        self.request_start = time.perf_counter()

    def first_token(self):

        now = time.perf_counter()

        self.prefill_end = now
        self.decode_start = now

    def token_generated(self):

        self.token_count += 1
        self.decode_end = time.perf_counter()

    def finish(self):

        total_latency = (self.decode_end - self.request_start) * 1000

        prefill_latency = (self.prefill_end - self.request_start) * 1000

        decode_time = (self.decode_end - self.decode_start)

        if self.token_count > 0:
            avg_token_latency = decode_time / self.token_count * 1000
            tokens_per_sec = self.token_count / decode_time
        else:
            avg_token_latency = 0
            tokens_per_sec = 0

        logger.info(
            "prefill=%.2f ms | token=%.2f ms | tok/s=%.2f | total=%.2f ms",
            prefill_latency,
            avg_token_latency,
            tokens_per_sec,
            total_latency
        )