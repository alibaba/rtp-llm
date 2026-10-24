package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import java.util.function.LongSupplier;

/** GenerateContext budgets. A retry deadline never truncates a live ALLOCATE RPC. */
record MockPdPolicy(long prefillRetryTimes, long prefillRetryTimeoutMs, long prefillMaxWaitTimeoutMs,
                    long decodeRetryIntervalMs, long loadCacheTimeoutMs, long maxRpcTimeoutMs,
                    int prepareThreads, long stopWaitTimeoutMs) {
    static final MockPdPolicy DEFAULT = new MockPdPolicy(0, 20, 600000, 1, 5000, 7200000, 128, 2000);
    static final String[] FIELDS = {"prefill_retry_times", "prefill_retry_timeout_ms", "prefill_max_wait_timeout_ms",
            "decode_retry_interval_ms", "load_cache_timeout_ms", "max_rpc_timeout_ms",
            "prefill_prepare_resource_pool_size", "prefill_stop_stream_wait_timeout_ms",
            "decode_retry_times", "decode_retry_timeout_ms"};
    static MockPdPolicy parse(JsonNode config) {
        MockEngineConfig.fields(config, FIELDS);
        for (String field : FIELDS)
            if (!config.path(field).isIntegralNumber() || !config.path(field).canConvertToLong())
                throw new IllegalArgumentException(field + " must be an int64");
        int threads = MockEngineConfig.integer(config, "prefill_prepare_resource_pool_size", 128);
        if (threads < 128) throw new IllegalArgumentException("prepare resource pool size must be real resolved value >=128");
        return new MockPdPolicy(config.path("prefill_retry_times").longValue(), config.path("prefill_retry_timeout_ms").longValue(),
                config.path("prefill_max_wait_timeout_ms").longValue(), config.path("decode_retry_interval_ms").longValue(),
                config.path("load_cache_timeout_ms").longValue(), config.path("max_rpc_timeout_ms").longValue(),
                threads, config.path("prefill_stop_stream_wait_timeout_ms").longValue());
    }
    long rpcMs(long remainingRequest) { return cap(remainingRequest, maxRpcTimeoutMs); }
    long loadMs(long remainingRequest) { return cap(rpcMs(remainingRequest), loadCacheTimeoutMs > 0 ? loadCacheTimeoutMs : 5000); }
    static long cap(long remaining, long relative) { return relative > 0 ? Math.min(remaining, relative) : remaining; }
    static long deadline(long started, long millis) {
        if (millis <= 0) return Long.MAX_VALUE;
        long nanos = TimeUnit.MILLISECONDS.toNanos(millis);
        return started > Long.MAX_VALUE - nanos ? Long.MAX_VALUE : started + nanos;
    }
    static long remaining(long deadline, long now) {
        return deadline == Long.MAX_VALUE ? Long.MAX_VALUE : Math.max(0, TimeUnit.NANOSECONDS.toMillis(deadline - now));
    }
    private static final ThreadLocal<Long> PREPARE_DEADLINE = ThreadLocal.withInitial(() -> Long.MAX_VALUE);
    long readyMs(long requestRemaining) { return Math.min(rpcMs(requestRemaining), remaining(PREPARE_DEADLINE.get(), System.nanoTime())); }
    interface Attempt { boolean run() throws Exception; }
    boolean prepare(Attempt attempt, BooleanSupplier cancelled, LongSupplier requestRemaining) throws Exception {
        long retryDeadline = deadline(System.nanoTime(), prefillRetryTimeoutMs);
        PREPARE_DEADLINE.set(retryDeadline);
        try { return prepareUntil(attempt, cancelled, requestRemaining, retryDeadline); }
        finally { PREPARE_DEADLINE.remove(); }
    }
    private boolean prepareUntil(Attempt attempt, BooleanSupplier cancelled, LongSupplier requestRemaining, long retryDeadline) throws Exception {
        for (long index = 0; ; index++) {
            if (cancelled.getAsBoolean()) throw MockRpcErrors.error(io.grpc.Status.CANCELLED, 8100, "prepare cancelled");
            if (requestRemaining.getAsLong() == 0) throw MockRpcErrors.error(io.grpc.Status.DEADLINE_EXCEEDED, 603, "request deadline exhausted during prepare");
            if (index > 0 && remaining(retryDeadline, System.nanoTime()) == 0) return false;
            boolean prepared = attempt.run();
            if (cancelled.getAsBoolean()) throw MockRpcErrors.error(io.grpc.Status.CANCELLED, 8100, "prepare cancelled");
            if (requestRemaining.getAsLong() == 0) throw MockRpcErrors.error(io.grpc.Status.DEADLINE_EXCEEDED, 603, "request deadline exhausted during prepare");
            if (prepared) return true;
            if (index >= Math.max(0, prefillRetryTimes) || remaining(retryDeadline, System.nanoTime()) == 0) return false;
            long sleep = Math.min(1, Math.min(requestRemaining.getAsLong(), remaining(retryDeadline, System.nanoTime())));
            if (sleep > 0) Thread.sleep(sleep);
        }
    }
}
