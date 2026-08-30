package org.flexlb.sync.lifecycle;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.slf4j.Logger;

import java.util.Map;
import java.util.Objects;

/**
 * The one production transaction for retiring a WorkerStatus generation.
 *
 * <p>This class stores no lifecycle state. {@link WorkerStatus} is the sole
 * owner of ACTIVE/RETIRING; this transaction only enforces the publication
 * order around that owner.
 */
public final class WorkerGenerationRetirement {

    private WorkerGenerationRetirement() {
    }

    /**
     * Close the exact endpoint gate, detach it from routing, and only then
     * publish RETIRING while the caller holds {@link WorkerStatus#lock}. The
     * status-map identity remains published throughout this transition.
     */
    public static EndpointRegistry.DetachedGeneration begin(
            WorkerStatus status,
            EndpointRegistry endpointRegistry,
            RoleType role,
            String ipPort) {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(ipPort, "ipPort");
        status.requireGenerationLock();
        status.requireActiveGeneration();

        EndpointRegistry.DetachedGeneration detached = null;
        if (endpointRegistry != null) {
            WorkerEndpoint expected = endpointRegistry.get(
                    role, ipPort, status);
            detached = endpointRegistry.detachAndBeginRetirement(
                    role, ipPort, status);
            if ((detached == null) != (expected == null)
                    || (detached != null
                        && !detached.ownsEndpoint(expected))) {
                throw new IllegalStateException(
                        "Exact endpoint detach invariant failed for "
                                + ipPort + "#" + status.getGenerationId());
            }
        }
        if (detached == null
                && !status.beginRetirementAfterEndpointGateClosed()) {
            throw new IllegalStateException(
                    "WorkerStatus generation changed while its lock was held: "
                            + ipPort + "#" + status.getGenerationId());
        }
        return detached;
    }

    /**
     * Drain the detached endpoint through its exact registry capability, then
     * remove the RETIRING business identity. Endpoint cleanup and callbacks
     * finish before status removal; exact cache cleanup remains best-effort.
     */
    public static void complete(
            WorkerStatus status,
            Map<String, WorkerStatus> statusMap,
            CacheAwareService cacheAwareService,
            String ipPort,
            EndpointRegistry.DetachedGeneration detachedGeneration,
            Logger logger) {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(statusMap, "statusMap");
        Objects.requireNonNull(cacheAwareService, "cacheAwareService");
        Objects.requireNonNull(ipPort, "ipPort");
        Objects.requireNonNull(logger, "logger");

        Throwable cleanupFailure = null;
        try {
            if (detachedGeneration != null) {
                detachedGeneration.retireAndAwait();
            }
        } catch (Throwable retirementFailure) {
            cleanupFailure = retirementFailure;
        } finally {
            // Business identity has one owner and one terminal action. The
            // detached endpoint barrier has resolved before this status can
            // disappear, even when endpoint cleanup failed.
            finalizeRetirement(
                    status,
                    statusMap,
                    cacheAwareService,
                    ipPort,
                    logger);
        }
        if (cleanupFailure != null) {
            logger.error(
                    "Endpoint cleanup failed after retiring generation {} for {}",
                    status.getGenerationId(), ipPort, cleanupFailure);
        }
    }

    /** No-fail terminal action invoked exactly once by this transaction. */
    private static void finalizeRetirement(
            WorkerStatus status,
            Map<String, WorkerStatus> statusMap,
            CacheAwareService cacheAwareService,
            String ipPort,
            Logger logger) {
        status.lock.lock();
        try {
            status.requireRetiringGeneration();
            if (statusMap.get(ipPort) != status) {
                logger.error(
                        "Status identity changed before retirement finalized for {}#{}",
                        ipPort, status.getGenerationId());
                return;
            }
            try {
                // The old identity is still published and replacement is
                // therefore fenced. Clear the address-only cache index before
                // allowing a new generation to reuse this address.
                cacheAwareService.removeEngineBlockCache(ipPort);
            } catch (Throwable cacheCleanupFailure) {
                logger.error(
                        "Cache cleanup failed while retiring generation {} for {}",
                        status.getGenerationId(), ipPort, cacheCleanupFailure);
            }
            boolean removed = statusMap.remove(ipPort, status);
            if (!removed) {
                logger.error(
                        "Exact status removal failed while its generation lock was held for {}#{}",
                        ipPort, status.getGenerationId());
                return;
            }
        } catch (Throwable finalizationFailure) {
            logger.error(
                    "Status retirement finalization failed for {}#{}",
                    ipPort, status.getGenerationId(), finalizationFailure);
            return;
        } finally {
            status.lock.unlock();
        }

    }
}
