import logging
import time

logger = logging.getLogger("vllm.v1.executor.multiproc_executor")
SLOW_WORKER_RPC_NS = 500_000_000


class VLLMWorkerDiagnostics:
    def __init__(self, worker):
        self.worker = worker
        self.started_ns = 0
        self.executed_ns = 0

    def execute(self, operation, args, kwargs):
        self.started_ns = time.perf_counter_ns()
        output = operation(*args, **kwargs)
        self.executed_ns = time.perf_counter_ns()
        return output

    def after_handoff(self, method, args):
        finished_ns = time.perf_counter_ns()
        if (
            method not in ("execute_model", "sample_tokens")
            or finished_ns - self.started_ns < SLOW_WORKER_RPC_NS
        ):
            return
        scheduler_output = args[0] if method == "execute_model" else None
        logger.warning(
            "[navi-vllm-slow-worker-rpc] method=%s rank=%d "
            "execute_ms=%.3f handoff_ms=%.3f "
            "started_perf_ns=%d handed_off_perf_ns=%d "
            "scheduled_requests=%d scheduled_tokens=%d",
            method,
            self.worker.rank,
            (self.executed_ns - self.started_ns) / 1_000_000,
            (finished_ns - self.executed_ns) / 1_000_000,
            self.started_ns,
            finished_ns,
            (
                len(scheduler_output.num_scheduled_tokens)
                if scheduler_output is not None
                else 0
            ),
            (
                scheduler_output.total_num_scheduled_tokens
                if scheduler_output is not None
                else 0
            ),
        )

    def enqueue_output(self, output, async_output_type):
        started_ns = time.perf_counter_ns()
        output_type = type(output).__name__
        if isinstance(output, async_output_type):
            output = output.get_output()
        prepared_ns = time.perf_counter_ns()
        status = self.worker.ResponseStatus
        result = (
            (status.FAILURE, str(output))
            if isinstance(output, Exception)
            else (status.SUCCESS, output)
        )
        if (response_mq := self.worker.worker_response_mq) is not None:
            response_mq.enqueue(result)
        finished_ns = time.perf_counter_ns()
        if finished_ns - started_ns >= SLOW_WORKER_RPC_NS:
            logger.warning(
                "[navi-vllm-slow-output-copy] rank=%d output_type=%s "
                "prepare_ms=%.3f enqueue_ms=%.3f "
                "started_perf_ns=%d finished_perf_ns=%d",
                self.worker.rank,
                output_type,
                (prepared_ns - started_ns) / 1_000_000,
                (finished_ns - prepared_ns) / 1_000_000,
                started_ns,
                finished_ns,
            )
