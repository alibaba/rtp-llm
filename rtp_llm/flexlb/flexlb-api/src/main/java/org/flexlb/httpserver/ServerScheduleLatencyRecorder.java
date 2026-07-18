package org.flexlb.httpserver;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.BalanceContext;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.LongAdder;

/**
 * Server-side Schedule latency recorder used by online evaluation.
 *
 * <p>The recorder uses fixed millisecond buckets and does not allocate on the
 * request path. Client transport and client-side channel queueing are excluded.
 */
@Slf4j
@Component
public class ServerScheduleLatencyRecorder {

    private static final int MAX_TRACKED_MS = 120_000;

    private volatile Window window = new Window();

    public void recordArrival(long arrivalNanos) {
        window.arrivals.record(arrivalNanos);
    }

    public void recordCompletion(BalanceContext context, long responseCompletedNanos) {
        if (context == null) {
            return;
        }
        Window current = window;
        current.completions.record(responseCompletedNanos);

        long grpcEntryNanos = context.getGrpcEntryNanos() > 0
                ? context.getGrpcEntryNanos() : context.getServiceStartNanos();
        current.serverTotal.recordBetween(grpcEntryNanos, responseCompletedNanos);
        current.grpcQueue.recordBetween(grpcEntryNanos, context.getServiceStartNanos());
        current.routeSubmit.recordBetween(context.getServiceStartNanos(), context.getRouteSubmittedNanos());
        current.batchWait.recordBetween(context.getRouteSubmittedNanos(), context.getBatchDispatchedNanos());
        current.dispatchAck.recordBetween(context.getBatchDispatchedNanos(), context.getAckAtNanos());
        current.ackResponse.recordBetween(context.getAckAtNanos(), responseCompletedNanos);
    }

    public Map<String, Object> snapshot() {
        Window current = window;
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("arrival_count", current.arrivals.count());
        result.put("arrival_qps", current.arrivals.qps());
        result.put("completion_count", current.completions.count());
        result.put("completion_qps", current.completions.qps());
        result.put("server_total_ms", current.serverTotal.snapshot());
        result.put("grpc_queue_ms", current.grpcQueue.snapshot());
        result.put("route_submit_ms", current.routeSubmit.snapshot());
        result.put("batch_wait_ms", current.batchWait.snapshot());
        result.put("dispatch_ack_ms", current.dispatchAck.snapshot());
        result.put("ack_response_ms", current.ackResponse.snapshot());
        return result;
    }

    public void reset() {
        window = new Window();
    }

    @Scheduled(fixedRateString = "${flexlb.server-latency-log-interval-ms:10000}")
    public void logSnapshot() {
        Window current = window;
        LatencySnapshot total = current.serverTotal.snapshotValue();
        if (total.count() == 0) {
            return;
        }
        log.info("flexlb_server_schedule_latency count={} arrival_qps={} completion_qps={} "
                        + "server_p50_ms={} server_p95_ms={} server_p99_ms={} "
                        + "grpc_queue_p95_ms={} route_submit_p95_ms={} batch_wait_p95_ms={} "
                        + "dispatch_ack_p95_ms={} ack_response_p95_ms={}",
                total.count(), current.arrivals.qps(), current.completions.qps(),
                total.p50(), total.p95(), total.p99(),
                current.grpcQueue.snapshotValue().p95(),
                current.routeSubmit.snapshotValue().p95(),
                current.batchWait.snapshotValue().p95(),
                current.dispatchAck.snapshotValue().p95(),
                current.ackResponse.snapshotValue().p95());
    }

    private static final class Window {
        private final WindowRate arrivals = new WindowRate();
        private final WindowRate completions = new WindowRate();
        private final LatencyHistogram serverTotal = new LatencyHistogram();
        private final LatencyHistogram grpcQueue = new LatencyHistogram();
        private final LatencyHistogram routeSubmit = new LatencyHistogram();
        private final LatencyHistogram batchWait = new LatencyHistogram();
        private final LatencyHistogram dispatchAck = new LatencyHistogram();
        private final LatencyHistogram ackResponse = new LatencyHistogram();
    }

    private static final class WindowRate {
        private final LongAdder count = new LongAdder();
        private final AtomicLong firstNanos = new AtomicLong(Long.MAX_VALUE);
        private final AtomicLong lastNanos = new AtomicLong(Long.MIN_VALUE);

        private void record(long timestampNanos) {
            count.increment();
            firstNanos.accumulateAndGet(timestampNanos, Math::min);
            lastNanos.accumulateAndGet(timestampNanos, Math::max);
        }

        private long count() {
            return count.sum();
        }

        private double qps() {
            long samples = count();
            long first = firstNanos.get();
            long last = lastNanos.get();
            if (samples < 2 || first == Long.MAX_VALUE || last <= first) {
                return 0.0;
            }
            return round3((samples - 1) * 1_000_000_000.0 / (last - first));
        }
    }

    private static final class LatencyHistogram {
        private final AtomicLongArray buckets = new AtomicLongArray(MAX_TRACKED_MS + 1);
        private final LongAdder count = new LongAdder();
        private final LongAdder sumMs = new LongAdder();
        private final AtomicLong maxMs = new AtomicLong();

        private void recordBetween(long startNanos, long endNanos) {
            if (startNanos <= 0 || endNanos < startNanos) {
                return;
            }
            long valueMs = TimeUnit.NANOSECONDS.toMillis(endNanos - startNanos);
            int bucket = (int) Math.min(valueMs, MAX_TRACKED_MS);
            buckets.incrementAndGet(bucket);
            count.increment();
            sumMs.add(valueMs);
            maxMs.accumulateAndGet(valueMs, Math::max);
        }

        private Map<String, Object> snapshot() {
            LatencySnapshot value = snapshotValue();
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("count", value.count());
            result.put("p50", value.p50());
            result.put("p90", value.p90());
            result.put("p95", value.p95());
            result.put("p99", value.p99());
            result.put("max", value.max());
            result.put("mean", value.mean());
            return result;
        }

        private LatencySnapshot snapshotValue() {
            long samples = count.sum();
            if (samples == 0) {
                return LatencySnapshot.EMPTY;
            }
            return new LatencySnapshot(
                    samples,
                    percentile(samples, 0.50),
                    percentile(samples, 0.90),
                    percentile(samples, 0.95),
                    percentile(samples, 0.99),
                    maxMs.get(),
                    round3(sumMs.sum() / (double) samples));
        }

        private long percentile(long samples, double percentile) {
            long target = Math.max(1, (long) Math.ceil(samples * percentile));
            long seen = 0;
            for (int value = 0; value <= MAX_TRACKED_MS; value++) {
                seen += buckets.get(value);
                if (seen >= target) {
                    return value;
                }
            }
            return MAX_TRACKED_MS;
        }
    }

    private record LatencySnapshot(long count, long p50, long p90, long p95,
                                   long p99, long max, double mean) {
        private static final LatencySnapshot EMPTY =
                new LatencySnapshot(0, 0, 0, 0, 0, 0, 0.0);
    }

    private static double round3(double value) {
        return Math.round(value * 1000.0) / 1000.0;
    }
}
