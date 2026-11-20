package org.flexlb.api;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.schedule.RequestContext;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.atomic.LongAdder;

/**
 * Server-side Schedule latency recorder used by online evaluation.
 *
 * <p>The recorder uses millisecond buckets with lazily allocated pages. Client transport and client-side channel queueing are excluded.
 */
@Slf4j
@Component
public class ServerScheduleLatencyRecorder {
    private static final int MAX_TRACKED_MS = 120_000;

    private volatile Window window = new Window();

    public void recordCompletion(RequestContext context, long responseCompletedNanos) {
        if (context == null) {
            return;
        }
        Window current = window;
        current.completions.record(responseCompletedNanos);

        long grpcEntryNanos =
                context.getTiming().grpcReceivedNanos() > 0
                        ? context.getTiming().grpcReceivedNanos()
                        : context.getTiming().startedNanos();
        current.serverTotal.recordBetween(grpcEntryNanos, responseCompletedNanos);
        current.grpcQueue.recordBetween(grpcEntryNanos, context.getTiming().startedNanos());
        current.routeSubmit.recordBetween(
                context.getTiming().startedNanos(), context.getTiming().routeCommitStartedNanos());
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
        log.info(
                "flexlb_server_schedule_latency count={} arrival_qps={} completion_qps={} "
                        + "server_p50_ms={} server_p95_ms={} server_p99_ms={} "
                        + "grpc_queue_p95_ms={} route_submit_p95_ms={}",
                total.count(),
                current.arrivals.qps(),
                current.completions.qps(),
                total.p50(),
                total.p95(),
                total.p99(),
                current.grpcQueue.snapshotValue().p95(),
                current.routeSubmit.snapshotValue().p95());
    }

    public void recordArrival(long arrivalNanos) {
        window.arrivals.record(arrivalNanos);
    }

    private static double round3(double value) {
        return Math.round(value * 1000.0) / 1000.0;
    }

    /**
     * Groups arrival, completion and stage-latency samples for one reporting window.
     */
    private static final class Window {
        private final WindowRate arrivals = new WindowRate();
        private final WindowRate completions = new WindowRate();
        private final LatencyHistogram serverTotal = new LatencyHistogram();
        private final LatencyHistogram grpcQueue = new LatencyHistogram();
        private final LatencyHistogram routeSubmit = new LatencyHistogram();
    }

    /**
     * Counts events and their observation interval to calculate the window's rate.
     */
    private static final class WindowRate {
        private final LongAdder count = new LongAdder();
        private final AtomicLong firstNanos = new AtomicLong(Long.MAX_VALUE);
        private final AtomicLong lastNanos = new AtomicLong(Long.MIN_VALUE);

        private void record(long timestampNanos) {
            count.increment();
            firstNanos.accumulateAndGet(timestampNanos, Math::min);
            lastNanos.accumulateAndGet(timestampNanos, Math::max);
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

        private long count() {
            return count.sum();
        }
    }

    /**
     * Accumulates millisecond samples and computes count, maximum and percentile snapshots.
     */
    private static final class LatencyHistogram {
        private static final int PAGE_SIZE = 256;
        private final AtomicReferenceArray<AtomicLongArray> pages =
                new AtomicReferenceArray<>(MAX_TRACKED_MS / PAGE_SIZE + 1);
        private final LongAdder count = new LongAdder();
        private final LongAdder sumMs = new LongAdder();
        private final AtomicLong maxMs = new AtomicLong();

        private LatencyHistogram() {
            pages.set(0, new AtomicLongArray(PAGE_SIZE));
        }

        private void recordBetween(long startNanos, long endNanos) {
            if (startNanos <= 0 || endNanos < startNanos) {
                return;
            }
            long valueMs = TimeUnit.NANOSECONDS.toMillis(endNanos - startNanos);
            int bucket = (int) Math.min(valueMs, MAX_TRACKED_MS);
            int pageIndex = bucket / PAGE_SIZE;
            AtomicLongArray page = pages.get(pageIndex);
            if (page == null) {
                AtomicLongArray fresh = new AtomicLongArray(PAGE_SIZE);
                page = pages.compareAndSet(pageIndex, null, fresh) ? fresh : pages.get(pageIndex);
            }
            page.incrementAndGet(bucket % PAGE_SIZE);
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
            for (int pageIndex = 0; pageIndex < pages.length(); pageIndex++) {
                AtomicLongArray page = pages.get(pageIndex);
                if (page == null) continue;
                for (int offset = 0; offset < PAGE_SIZE; offset++) {
                    seen += page.get(offset);
                    if (seen >= target) return pageIndex * PAGE_SIZE + offset;
                }
            }
            return MAX_TRACKED_MS;
        }
    }

    /**
     * Freezes latency statistics so reporting does not retain mutable histogram buckets.
     */
    private record LatencySnapshot(
            long count, long p50, long p90, long p95, long p99, long max, double mean) {
        private static final LatencySnapshot EMPTY = new LatencySnapshot(0, 0, 0, 0, 0, 0, 0.0);
    }
}
