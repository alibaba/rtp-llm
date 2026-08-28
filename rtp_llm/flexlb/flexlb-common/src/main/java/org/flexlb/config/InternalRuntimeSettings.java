package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;

import java.util.Locale;
import java.util.Map;

/**
 * Implementation sizing defaults. These values are intentionally not part of
 * the public FLEXLB_CONFIG schema.
 *
 * <p>The batch dispatch sizing is additionally overridable through process
 * environment variables ({@code FLEXLB_BATCH_DISPATCH_THREADS} and
 * {@code FLEXLB_BATCH_DISPATCH_QUEUE_CAPACITY}) so overload benchmarks can
 * widen the dispatcher admission gate without editing code. Defaults keep the
 * upstream production sizing (64 threads + 256 queue slots = 320 admission
 * permits). Invalid or missing values fall back to the defaults.</p>
 *
 * <p>The route admission block projection is likewise switchable through
 * {@code FLEXLB_ROUTE_ADMISSION_BLOCK_PROJECTION} ("0"/"false" disables):
 * with the projection off, worker snapshots carry no admission card, so
 * RouteAdmissionPolicy never marks an endpoint BLOCKED from an observed
 * admission wait and requests queue instead of being fast-rejected. The
 * default (enabled) keeps production behavior byte-identical.</p>
 */
@Getter
public final class InternalRuntimeSettings {

    private final int prefillSaturatedAtPendingRequests = 20;
    private final long decodeFullSpeedBelowKvUsagePercent = 40;
    private final long decodeSaturatedAtKvUsagePercent = 80;

    private final int nettySelectThreadMultiplier = 1;
    private final int nettyWorkerThreadMultiplier = 2;
    private final int grpcClientExecutorThreads = 32;
    private final int grpcClientExecutorQueueCapacity = 10_000;
    private final int grpcClientEventLoopThreads = 8;
    private final int grpcServerWorkerEventLoopThreads = 4;
    private final int httpNettyEventLoopThreads = 4;
    private final int httpNettyEventExecutorThreads = 16;
    private final int httpNettyEventExecutorQueueCapacity = 1000;
    private final int httpRequestExecutorThreads = 32;
    private final int httpRequestExecutorQueueCapacity = 10_000;
    private final int engineSyncExecutorThreads = 32;
    private final int statusCheckExecutorThreads = 32;
    private final int serviceDiscoveryMaxThreads = 32;
    private final int batchDispatchThreads;
    private final int batchDispatchQueueCapacity;
    /**
     * Route admission block projection switch. Default true (production):
     * a parked admission head's wait is projected onto the endpoint snapshot
     * so RouteAdmissionPolicy can BLOCK the endpoint (fast-reject form).
     * False (env "0"/"false") removes the card from every snapshot: the
     * endpoint stays selectable and excess requests queue instead. The
     * dispatch-side park logic itself is NOT affected either way.
     */
    @JsonIgnore
    private final boolean routeAdmissionBlockProjectionEnabled;
    private final int fallbackBatchTokenCapacity = 1_048_576;
    private final long masterForwardRpcTimeoutMs = 5000;

    static final int DEFAULT_BATCH_DISPATCH_THREADS = 64;
    static final int DEFAULT_BATCH_DISPATCH_QUEUE_CAPACITY = 256;
    static final String BATCH_DISPATCH_THREADS_ENV = "FLEXLB_BATCH_DISPATCH_THREADS";
    static final String BATCH_DISPATCH_QUEUE_CAPACITY_ENV = "FLEXLB_BATCH_DISPATCH_QUEUE_CAPACITY";
    static final String ROUTE_ADMISSION_BLOCK_PROJECTION_ENV =
            "FLEXLB_ROUTE_ADMISSION_BLOCK_PROJECTION";

    public InternalRuntimeSettings() {
        this(System.getenv());
    }

    /** Package-visible env injection keeps unit tests deterministic. */
    InternalRuntimeSettings(Map<String, String> env) {
        this.batchDispatchThreads = positiveIntOrDefault(
                env.get(BATCH_DISPATCH_THREADS_ENV), DEFAULT_BATCH_DISPATCH_THREADS);
        this.batchDispatchQueueCapacity = positiveIntOrDefault(
                env.get(BATCH_DISPATCH_QUEUE_CAPACITY_ENV), DEFAULT_BATCH_DISPATCH_QUEUE_CAPACITY);
        this.routeAdmissionBlockProjectionEnabled = flagOrDefault(
                env.get(ROUTE_ADMISSION_BLOCK_PROJECTION_ENV), true);
    }

    private static int positiveIntOrDefault(String raw, int fallback) {
        if (raw == null || raw.isEmpty()) {
            return fallback;
        }
        try {
            int parsed = Integer.parseInt(raw.trim());
            return parsed > 0 ? parsed : fallback;
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    /**
     * Boolean env parsing: explicit "0"/"false" (case- and whitespace-insensitive)
     * disables, explicit "1"/"true" enables, anything else (missing, blank,
     * unrecognized) falls back to the default so typos can never silently flip
     * production behavior.
     */
    private static boolean flagOrDefault(String raw, boolean fallback) {
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        String normalized = raw.trim().toLowerCase(Locale.ROOT);
        if ("0".equals(normalized) || "false".equals(normalized)) {
            return false;
        }
        if ("1".equals(normalized) || "true".equals(normalized)) {
            return true;
        }
        return fallback;
    }
}
