package org.flexlb.cache.match.localstandby;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Coordinates Local Standby cache matching, request-derived metadata updates and index sizing.
 *
 * <p>The underlying metadata is approximate and is not replicated between FlexLB master and
 * follower.
 */
@Component
@Slf4j
public class LocalStandbyCacheManager {

    private static final long CAPACITY_REFRESH_INTERVAL_MS = 60_000;
    private static final long CAPACITY_WARNING_INTERVAL_NANOS = TimeUnit.MINUTES.toNanos(1);
    private final boolean enabled;
    private final WorkerStatusProvider workerStatusProvider;
    private final CacheMetricsReporter cacheMetricsReporter;
    private final Collection<ServiceRoute> serviceRoutes;
    private final long hardMaximumEntries;
    private final double capacityMultiplier;
    private final long configuredBlockSize;
    private final LocalStandbyCacheIndex cacheIndex;
    private volatile long nextCapacityWarningNanos;

    public LocalStandbyCacheManager(CacheMatchConfiguration configuration, WorkerStatusProvider workerStatusProvider, CacheMetricsReporter cacheMetricsReporter) {
        LocalStandbyConfig config = configuration.getLocalStandbyConfig();
        this.enabled = configuration.isLocalStandbyEnabled();
        this.workerStatusProvider = workerStatusProvider;
        this.cacheMetricsReporter = cacheMetricsReporter;
        this.serviceRoutes = configuration.getServiceRoutes();
        this.hardMaximumEntries = enabled
                ? config.getMaximumEntries()
                : LocalStandbyConfig.DEFAULT_MAXIMUM_ENTRIES;
        this.capacityMultiplier = enabled
                ? config.getCapacityMultiplier()
                : LocalStandbyConfig.DEFAULT_CAPACITY_MULTIPLIER;
        this.configuredBlockSize = enabled ? config.getBlockSize() : 0;
        long entryTtlMs = enabled
                ? config.getEntryTtlMs()
                : LocalStandbyConfig.DEFAULT_ENTRY_TTL_MS;
        this.cacheIndex = new LocalStandbyCacheIndex(entryTtlMs, hardMaximumEntries, enabled);
    }

    public Map<String, Integer> findMatchingEngines(List<Long> blockCacheKeys, RoleType roleType, String group) {
        if (!enabled || blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyMap();
        }

        Collection<WorkerStatus> workerStatuses = workerStatusProvider.getWorkerStatuses(roleType, group);
        if (workerStatuses == null || workerStatuses.isEmpty()) {
            return Collections.emptyMap();
        }
        Map<String, Integer> prefixMatches = calculatePrefixMatchBlockCounts(blockCacheKeys, workerStatuses);
        applyCacheMatchRollback(prefixMatches, workerStatuses);
        return prefixMatches;
    }

    private Map<String, Integer> calculatePrefixMatchBlockCounts(List<Long> blockCacheKeys, Collection<WorkerStatus> workerStatuses) {
        // A worker remains a candidate until the first block it does not own or whose mapping has expired.
        Set<String> candidateWorkers = new LinkedHashSet<>(workerStatuses.size());
        for (WorkerStatus workerStatus : workerStatuses) {
            if (workerStatus == null) {
                continue;
            }
            String workerIpPort = workerStatus.getIpPort();
            if (StringUtils.isNotBlank(workerIpPort)) {
                candidateWorkers.add(workerIpPort);
            }
        }
        if (candidateWorkers.isEmpty()) {
            return Collections.emptyMap();
        }

        Map<String, Integer> prefixMatches = new LinkedHashMap<>(candidateWorkers.size());
        long queryTimeNanos = System.nanoTime();
        // Check blocks in order because a miss interrupts the worker's contiguous prefix match.
        for (int blockIndex = 0; blockIndex < blockCacheKeys.size(); blockIndex++) {
            Long blockCacheKey = blockCacheKeys.get(blockIndex);
            Map<String, Long> blockOwners = cacheIndex.getUnexpiredEnginesForBlock(blockCacheKey, queryTimeNanos);

            Iterator<String> candidateIterator = candidateWorkers.iterator();
            while (candidateIterator.hasNext()) {
                String candidateWorker = candidateIterator.next();
                if (blockOwners == null) {
                    // No worker owns the current block, so every remaining prefix ends here.
                    prefixMatches.put(candidateWorker, blockIndex);
                    candidateIterator.remove();
                    continue;
                }

                if (!blockOwners.containsKey(candidateWorker)) {
                    // The worker does not own the current block, so its prefix ends here.
                    prefixMatches.put(candidateWorker, blockIndex);
                    candidateIterator.remove();
                }
            }

            // Later blocks cannot restore an interrupted prefix.
            if (candidateWorkers.isEmpty()) {
                break;
            }
        }

        // Workers still present matched every requested block.
        for (String candidateWorker : candidateWorkers) {
            prefixMatches.put(candidateWorker, blockCacheKeys.size());
        }
        return prefixMatches;
    }

    private void applyCacheMatchRollback(Map<String, Integer> prefixMatches,
                                         Collection<WorkerStatus> workerStatuses) {
        for (WorkerStatus workerStatus : workerStatuses) {
            if (workerStatus == null || workerStatus.getCacheMatchRollbackBlocks() <= 0) {
                continue;
            }
            String workerIpPort = workerStatus.getIpPort();
            Integer matchedBlocks = prefixMatches.get(workerIpPort);
            if (matchedBlocks != null) {
                prefixMatches.put(workerIpPort,
                        Math.max(matchedBlocks - workerStatus.getCacheMatchRollbackBlocks(), 0));
            }
        }
    }

    public void addRoutedRequestBlocks(String workerIpPort, List<Long> blockCacheKeys) {
        int rejectedMappings = cacheIndex.addWorkerBlockMappings(workerIpPort, blockCacheKeys);
        if (rejectedMappings > 0) {
            cacheMetricsReporter.reportLocalStandbyCapacityRejected();
            long now = System.nanoTime();
            if (now >= nextCapacityWarningNanos) {
                nextCapacityWarningNanos = now + CAPACITY_WARNING_INTERVAL_NANOS;
                log.warn("Local Standby cache capacity reached; rejected {} new mappings, "
                                + "currentMappings={}, maximumEntries={} (warning limited to once per minute)",
                        rejectedMappings, cacheIndex.mappingCount(), cacheIndex.maximumEntryCount());
            }
        }
    }

    public long mappingCount() {
        return cacheIndex.mappingCount();
    }

    public long maximumEntryCount() {
        return cacheIndex.maximumEntryCount();
    }

    @Scheduled(fixedRate = 20_000)
    public void reportMappingCount() {
        if (enabled) {
            cacheMetricsReporter.reportLocalStandbyMappingCount(cacheIndex.mappingCount());
        }
    }

    @Scheduled(fixedDelay = CAPACITY_REFRESH_INTERVAL_MS)
    void refreshMaximumEntries() {
        if (!enabled) {
            return;
        }

        try {
            long estimatedHbmBlockCapacity = estimateHbmBlockCapacity();
            if (estimatedHbmBlockCapacity <= 0) {
                return;
            }

            double capacityWithHeadroom = estimatedHbmBlockCapacity * capacityMultiplier;
            long newMaximum = capacityWithHeadroom >= hardMaximumEntries
                    ? hardMaximumEntries
                    : Math.max(1, (long) Math.ceil(capacityWithHeadroom));
            long previousMaximum = cacheIndex.maximumEntryCount();
            if (newMaximum == previousMaximum) {
                return;
            }

            cacheIndex.updateMaximumEntries(newMaximum);
            log.info("Updated local standby cache capacity from {} to {} entries (estimatedHbmBlocks={}, multiplier={}, hardMaximum={})",
                    previousMaximum,
                    newMaximum,
                    estimatedHbmBlockCapacity,
                    capacityMultiplier,
                    hardMaximumEntries);
        } catch (RuntimeException e) {
            log.warn("Failed to update local standby cache capacity; keeping {} entries", cacheIndex.maximumEntryCount(), e);
        }
    }

    private long estimateHbmBlockCapacity() {
        long totalCapacity = 0;
        Set<WorkerCapacityKey> countedWorkers = new HashSet<>();
        for (ServiceRoute serviceRoute : serviceRoutes) {
            for (RoleType roleType : serviceRoute.getAllRoleTypes()) {
                totalCapacity = addRoleCapacity(totalCapacity, countedWorkers, serviceRoute, roleType);
                if (totalCapacity >= hardMaximumEntries) {
                    return hardMaximumEntries;
                }
            }
        }
        return totalCapacity;
    }

    private long addRoleCapacity(long currentCapacity, Set<WorkerCapacityKey> countedWorkers, ServiceRoute serviceRoute, RoleType roleType) {
        long totalCapacity = currentCapacity;
        for (var endpointWithGroup : serviceRoute.getAllEndpointsWithGroup(roleType)) {
            if (endpointWithGroup.getRight() == null) {
                continue;
            }
            String group = StringUtils.defaultString(endpointWithGroup.getLeft());
            Collection<WorkerStatus> workerStatuses = workerStatusProvider.getWorkerStatuses(roleType, group);
            if (workerStatuses == null) {
                continue;
            }
            for (WorkerStatus workerStatus : workerStatuses) {
                long workerCapacity = calculateWorkerBlockCapacity(workerStatus);
                if (workerCapacity <= 0) {
                    continue;
                }
                WorkerCapacityKey workerKey = new WorkerCapacityKey(group, roleType, workerStatus.getIpPort());
                if (!countedWorkers.add(workerKey)) {
                    continue;
                }
                totalCapacity += workerCapacity;
                if (totalCapacity >= hardMaximumEntries) {
                    return hardMaximumEntries;
                }
            }
        }
        return totalCapacity;
    }

    private long calculateWorkerBlockCapacity(WorkerStatus workerStatus) {
        if (workerStatus == null || !workerStatus.isAlive() || workerStatus.getCacheStatus() == null) {
            return 0;
        }

        var cacheStatus = workerStatus.getCacheStatus();
        if (cacheStatus.getTotalKvCache() <= 0 || cacheStatus.getBlockSize() <= 0) {
            return 0;
        }

        long blockSize = configuredBlockSize > 0 ? configuredBlockSize : cacheStatus.getBlockSize();
        return divideRoundUp(cacheStatus.getTotalKvCache(), blockSize);
    }

    private long divideRoundUp(long value, long divisor) {
        return value / divisor + (value % divisor == 0 ? 0 : 1);
    }

    @PreDestroy
    public void shutdown() {
        cacheIndex.shutdown();
    }

    private record WorkerCapacityKey(String group, RoleType roleType, String workerIpPort) {
    }
}
