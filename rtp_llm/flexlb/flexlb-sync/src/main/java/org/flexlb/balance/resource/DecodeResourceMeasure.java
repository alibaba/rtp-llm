package org.flexlb.balance.resource;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * Decode role resource measure
 * Availability criteria: KV cache usage percentage below threshold and decode concurrency below limit
 *
 * <p><b>Stage-2 T7 S2.5 dual-view gate semantics (ruling 4):</b>
 * {@link #isResourceAvailable} deliberately mixes two KV-views of the same
 * endpoint — the split is the contract, not an accident. The concurrency
 * gate reads the <b>engineFacing</b> view while the KV-usage gate reads the
 * <b>real hard</b> view; see the per-gate comments below for the ownership
 * of each reading.
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Component
public class DecodeResourceMeasure implements ResourceMeasure {
    private final long availableThreshold;
    private final long fullSpeedThreshold;
    private final long stopThreshold;
    private final long concurrencyLimit;

    public DecodeResourceMeasure(ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.availableThreshold = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxKvUsagePercent();
        this.fullSpeedThreshold = config.getInternalRuntime()
                .getDecodeFullSpeedBelowKvUsagePercent();
        this.stopThreshold = config.getInternalRuntime()
                .getDecodeSaturatedAtKvUsagePercent();
        Long configuredLimit = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests();
        this.concurrencyLimit = configuredLimit == null ? 0 : configuredLimit;
    }

    /**
     * Pure availability decision over one caller-owned routing snapshot.
     *
     * <p><b>Dual-view gate semantics (stage-2 T7 S2.5, ruling 4):</b> the
     * two gates below read different KV-views of the same endpoint, and
     * the split is deliberate:
     * <ul>
     *   <li><b>Concurrency gate — engineFacing semantics</b>
     *       ({@code view.engineLoad()}, derived from
     *       {@link DecodeEndpoint#getEngineLoad()}): confirmed running
     *       requests plus non-queued reservations plus dispatch permits.
     *       Reservations parked in a prefill queue guard KV only and must
     *       not saturate this limit (root cause C of the 8400 storm:
     *       shadow saturation on an idle engine).</li>
     *   <li><b>KV-usage gate — real hard semantics</b>
     *       ({@code view.realKvUsed()} over {@code view.totalKv()}): the
     *       conservative full-accounting view — engine-reported used plus
     *       every local reservation <b>including the queued soft holds</b>
     *       plus the priority-claim / engine-fence retained expected
     *       demand (see {@link DecodeEndpoint#realKvUsed()}). Queued work
     *       will eventually demand KV, so placement availability must
     *       stay conservative even though the concurrency gate above does
     *       not count it.</li>
     * </ul>
     */
    public boolean isResourceAvailable(
            DecodeEndpoint.DecodeRoutingView view) {
        if (view == null) {
            return false;
        }
        // Gate 1 — engineFacing semantics: confirmed + non-queued
        // reservations + dispatch permits. Queued reservations are
        // excluded: they guard KV only and must not saturate the engine
        // concurrency limit (root cause C of the 8400 storm: shadow
        // saturation on an idle engine).
        long engineLoad = view.engineLoad();
        if (concurrencyLimit > 0 && engineLoad >= concurrencyLimit) {
            return false;
        }
        // Gate 2 — real hard semantics: engine-reported used plus every
        // local reservation (queued soft holds included) plus the
        // priority/fence retained expected demand — the conservative
        // placement view, so queued work cannot silently oversubscribe KV.
        long used = view.realKvUsed();
        long total = view.totalKv();
        if (total == 0) {
            return true;
        }
        double usagePercentage = used * 100.0 / total;
        return usagePercentage < availableThreshold;
    }

    @Override
    public ResourceMeasureIndicatorEnum getResourceMeasureIndicator() {
        return ResourceMeasureIndicatorEnum.REMAINING_KV_CACHE;
    }

    @Override
    public double calculateAverageWaterLevel(Map<String, WorkerStatus> workerStatusMap) {
        if (MapUtils.isEmpty(workerStatusMap)) {
            return 0.0;
        }

        double totalWaterLevel = 0;
        int count = 0;

        for (WorkerStatus worker : workerStatusMap.values()) {
            double waterLevel = calculateWaterLevel(worker);
            totalWaterLevel += waterLevel;
            count++;
        }

        return count > 0 ? totalWaterLevel / count : 0.0;
    }

    private double calculateWaterLevel(WorkerStatus workerStatus) {
        if (workerStatus == null) {
            return 0.0;
        }
        WorkerStatus.EngineObservation status =
                workerStatus.committedEngineObservation();
        return Math.max(calculateKvCacheWaterLevel(status),
                calculateConcurrencyWaterLevel(status));
    }

    private double calculateKvCacheWaterLevel(
            WorkerStatus.EngineObservation status) {
        long total = status.totalKvCacheTokens();
        long available = status.availableKvCacheTokens();
        long used = total - available;

        if (total == 0) {
            return 0.0;
        }

        double usedPercentage = (used * 100.0) / total;

        if (usedPercentage <= fullSpeedThreshold) {
            return 0.0;
        } else if (usedPercentage >= stopThreshold) {
            return 100.0;
        } else {
            return (usedPercentage - fullSpeedThreshold) /
                    (stopThreshold - fullSpeedThreshold) * 100.0;
        }
    }

    private double calculateConcurrencyWaterLevel(
            WorkerStatus.EngineObservation status) {
        if (concurrencyLimit <= 0) {
            return 0.0;
        }

        long currentConcurrency = calculateDecodeConcurrency(status);
        if (currentConcurrency <= 0) {
            return 0.0;
        }
        return Math.min(100.0, currentConcurrency * 100.0 / concurrencyLimit);
    }

    private long calculateDecodeConcurrency(
            WorkerStatus.EngineObservation status) {
        if (MapUtils.isNotEmpty(status.runningTaskList())) {
            return status.runningTaskList().size();
        }
        return 0;
    }
}
