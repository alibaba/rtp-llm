package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.Objects;
import java.util.concurrent.ConcurrentMap;
import java.util.function.Predicate;

/**
 * Owns WorkerStatus generation creation, replacement and retirement.
 * Endpoint and cache cleanup are conditional on the exact WorkerStatus object.
 */
@Component
public class WorkerGenerationManager {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final EndpointRegistry endpointRegistry;
    private final CacheAwareService cacheAwareService;
    private final WorkerGenerationFence generationFence;

    public WorkerGenerationManager(EndpointRegistry endpointRegistry,
                                   CacheAwareService cacheAwareService,
                                   WorkerGenerationFence generationFence) {
        this.endpointRegistry = endpointRegistry;
        this.cacheAwareService = cacheAwareService;
        this.generationFence = generationFence;
    }

    public WorkerStatus getOrCreate(ConcurrentMap<String, WorkerStatus> statuses,
                                    RoleType role, String ipPort) {
        return statuses.computeIfAbsent(ipPort, ignored -> newGeneration(role, ipPort));
    }

    public boolean isCurrent(ConcurrentMap<String, WorkerStatus> statuses,
                             String ipPort, WorkerStatus expected) {
        return statuses.get(ipPort) == expected;
    }

    public boolean retireIf(ConcurrentMap<String, WorkerStatus> statuses,
                            RoleType role, String ipPort, WorkerStatus expected,
                            Predicate<WorkerStatus> eligibility) {
        return generationFence.write(ipPort, () -> {
            WorkerStatus current = statuses.get(ipPort);
            if (current != expected || !eligibility.test(current)) {
                return false;
            }
            // Stop queued old-generation work before removing the generation
            // identity. This hook is short and does not hold a CHM bin lock.
            endpointRegistry.beginEndpointRetirement(role, ipPort, current);
            current.setAlive(false);
            if (!statuses.remove(ipPort, current)) {
                return false;
            }
            return finishRetirement(role, ipPort, current);
        });
    }

    /**
     * Replace an address generation after a strictly lower status version is
     * observed. The triggering response is discarded; the next poll starts
     * from a fresh finished cursor.
     */
    public boolean rotateOnVersionRollback(ConcurrentMap<String, WorkerStatus> statuses,
                                           RoleType role, String ipPort,
                                           WorkerStatus expected,
                                           long observedVersion) {
        return generationFence.write(ipPort, () -> {
            WorkerStatus current = statuses.get(ipPort);
            if (current != expected
                    || observedVersion >= current.getStatusVersion().get()) {
                return false;
            }
            WorkerStatus replacement = newGeneration(role, ipPort);
            endpointRegistry.beginEndpointRetirement(role, ipPort, current);
            current.setAlive(false);
            if (!statuses.replace(ipPort, current, replacement)) {
                return false;
            }
            return finishRetirement(role, ipPort, current);
        });
    }

    private boolean finishRetirement(RoleType role, String ipPort, WorkerStatus retired) {
        if (retired == null) {
            return false;
        }
        try {
            endpointRegistry.remove(role, ipPort, retired);
        } catch (RuntimeException endpointFailure) {
            logger.error("Failed to retire endpoint for worker={}, role={}",
                    ipPort, role, endpointFailure);
        }
        if (role == RoleType.PREFILL || role == RoleType.PDFUSION) {
            try {
                cacheAwareService.clearEngineCache(ipPort);
            } catch (RuntimeException cacheFailure) {
                logger.error("Failed to clear cache for retired worker={}, role={}",
                        ipPort, role, cacheFailure);
            }
        }
        return true;
    }

    private static WorkerStatus newGeneration(RoleType role, String ipPort) {
        Objects.requireNonNull(role, "role");
        int separator = ipPort.lastIndexOf(':');
        if (separator <= 0 || separator == ipPort.length() - 1) {
            throw new IllegalArgumentException("Invalid worker address: " + ipPort);
        }
        String ip = ipPort.substring(0, separator);
        int port = Integer.parseInt(ipPort.substring(separator + 1));

        WorkerStatus status = new WorkerStatus();
        status.setRole(role);
        status.setIp(ip);
        status.setPort(port);
        status.setGrpcPort(CommonUtils.toGrpcPort(port));
        status.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
        return status;
    }
}
