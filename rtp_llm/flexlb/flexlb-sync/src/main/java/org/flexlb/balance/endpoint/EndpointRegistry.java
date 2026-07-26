package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.ObjectFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

@Component
public class EndpointRegistry {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final ConcurrentHashMap<String, PrefillEndpoint> prefillEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, PrefillEndpoint> pdFusionEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, SimpleWorkerEndpoint> vitEndpoints = new ConcurrentHashMap<>();
    private final ConfigService configService;
    private final ObjectFactory<FlexlbBatchScheduler> batchSchedulerFactory;
    private final BatchSchedulerReporter reporter;

    public EndpointRegistry(ConfigService configService,
                            ObjectFactory<FlexlbBatchScheduler> batchSchedulerFactory,
                            BatchSchedulerReporter reporter) {
        this.configService = configService;
        this.batchSchedulerFactory = batchSchedulerFactory;
        this.reporter = reporter;
    }

    public WorkerEndpoint get(RoleType roleType, String ipPort) {
        if (roleType == RoleType.PREFILL) {
            return getPrefill(ipPort);
        }
        if (roleType == RoleType.DECODE) {
            return getDecode(ipPort);
        }
        if (roleType == RoleType.PDFUSION) {
            return getPdFusion(ipPort);
        }
        if (roleType == RoleType.VIT) {
            return getVit(ipPort);
        }
        return null;
    }

    public Map<String, ? extends WorkerEndpoint> getEndpoints(RoleType roleType) {
        if (roleType == RoleType.PREFILL) {
            return prefillEndpoints;
        }
        if (roleType == RoleType.DECODE) {
            return decodeEndpoints;
        }
        if (roleType == RoleType.PDFUSION) {
            return pdFusionEndpoints;
        }
        if (roleType == RoleType.VIT) {
            return vitEndpoints;
        }
        return Map.of();
    }

    public PrefillEndpoint getPrefill(String ipPort) {
        return prefillEndpoints.get(ipPort);
    }

    public DecodeEndpoint getDecode(String ipPort) {
        return decodeEndpoints.get(ipPort);
    }

    private PrefillEndpoint getPdFusion(String ipPort) {
        return pdFusionEndpoints.get(ipPort);
    }

    private SimpleWorkerEndpoint getVit(String ipPort) {
        return vitEndpoints.get(ipPort);
    }

    public WorkerEndpoint ensureEndpoint(RoleType roleType, String ipPort, WorkerStatus status) {
        if (roleType == RoleType.PREFILL) {
            return ensurePrefillEndpoint(ipPort, status, roleType);
        }
        if (roleType == RoleType.DECODE) {
            return ensureDecodeEndpoint(ipPort, status);
        }
        if (roleType == RoleType.PDFUSION) {
            return ensurePdFusionEndpoint(ipPort, status, roleType);
        }
        if (roleType == RoleType.VIT) {
            return ensureVitEndpoint(ipPort, status);
        }
        throw new IllegalArgumentException("Unsupported role: " + roleType);
    }

    private PrefillEndpoint ensurePrefillEndpoint(String ipPort, WorkerStatus status, RoleType roleType) {
        PrefillEndpoint endpoint = prefillEndpoints.get(ipPort);
        if (endpoint != null && endpoint.getStatus() == status) {
            return endpoint;
        }
        return ensureEndpoint(prefillEndpoints, ipPort, status,
                candidateStatus -> createPrefillEndpoint(candidateStatus, roleType, ipPort));
    }

    private DecodeEndpoint ensureDecodeEndpoint(String ipPort, WorkerStatus status) {
        return ensureEndpoint(decodeEndpoints, ipPort, status,
                candidateStatus -> createDecodeEndpoint(candidateStatus, ipPort));
    }

    private PrefillEndpoint ensurePdFusionEndpoint(String ipPort, WorkerStatus status, RoleType roleType) {
        PrefillEndpoint endpoint = pdFusionEndpoints.get(ipPort);
        if (endpoint != null && endpoint.getStatus() == status) {
            return endpoint;
        }
        return ensureEndpoint(pdFusionEndpoints, ipPort, status,
                candidateStatus -> createPrefillEndpoint(candidateStatus, roleType, ipPort));
    }

    private SimpleWorkerEndpoint ensureVitEndpoint(String ipPort, WorkerStatus status) {
        return ensureEndpoint(vitEndpoints, ipPort, status,
                candidateStatus -> createSimpleEndpoint(candidateStatus, RoleType.VIT, ipPort));
    }

    private <T extends WorkerEndpoint> T ensureEndpoint(ConcurrentHashMap<String, T> endpoints,
                                                         String ipPort,
                                                         WorkerStatus status,
                                                         Function<WorkerStatus, T> factory) {
        T current = endpoints.get(ipPort);
        if (current != null && current.getStatus() == status) {
            return current;
        }

        T candidate = factory.apply(status);
        while (true) {
            if (current == null) {
                T raced = endpoints.putIfAbsent(ipPort, candidate);
                if (raced == null) {
                    return candidate;
                }
                current = raced;
            }

            if (current.getStatus() == status) {
                candidate.close();
                return current;
            }
            if (endpoints.replace(ipPort, current, candidate)) {
                current.close();
                return candidate;
            }
            current = endpoints.get(ipPort);
        }
    }

    /**
     * Remove an endpoint only if it still belongs to the expired status generation.
     */
    public boolean remove(RoleType roleType, String ipPort, WorkerStatus expectedStatus) {
        if (expectedStatus == null) {
            return false;
        }
        expectedStatus.setAlive(false);
        if (roleType == RoleType.PREFILL) {
            return remove(prefillEndpoints, ipPort, expectedStatus);
        }
        if (roleType == RoleType.DECODE) {
            return remove(decodeEndpoints, ipPort, expectedStatus);
        }
        if (roleType == RoleType.PDFUSION) {
            return remove(pdFusionEndpoints, ipPort, expectedStatus);
        }
        if (roleType == RoleType.VIT) {
            return remove(vitEndpoints, ipPort, expectedStatus);
        }
        return false;
    }

    private <T extends WorkerEndpoint> boolean remove(ConcurrentHashMap<String, T> endpoints,
                                                       String ipPort,
                                                       WorkerStatus expectedStatus) {
        T endpoint = endpoints.get(ipPort);
        if (endpoint == null || endpoint.getStatus() != expectedStatus
                || !endpoints.remove(ipPort, endpoint)) {
            return false;
        }
        logOrphanInflight(ipPort, endpoint);
        endpoint.close();
        return true;
    }

    /**
     * Defect #5 diagnostics: when a worker is removed (e.g. marked dead after
     * consecutive gRPC failures) while it still tracks inflight entries, those
     * entries are orphaned and can only be reclaimed by TTL. Log the details so
     * the leak source is directly visible.
     */
    private void logOrphanInflight(String ipPort, WorkerEndpoint endpoint) {
        RoleType schedulerRole = null;
        if (endpoint instanceof PrefillEndpoint prefillEp) {
            schedulerRole = RoleType.PREFILL;
            int orphanBatches = prefillEp.getInflightBatchCount();
            if (orphanBatches > 0) {
                logger.warn("Endpoint removed with orphan inflight: role=PREFILL endpoint={} orphan_batch_count={} "
                                + "sample_batch_ids={} — entries reclaimable only by TTL",
                        ipPort, orphanBatches, prefillEp.sampleInflightBatchIds(10));
            }
        } else if (endpoint instanceof DecodeEndpoint decodeEp) {
            schedulerRole = RoleType.DECODE;
            int orphanRequests = decodeEp.getInflightCount();
            if (orphanRequests > 0) {
                logger.warn("Endpoint removed with orphan inflight: role=DECODE endpoint={} orphan_request_count={} "
                                + "sample_request_ids={} — entries reclaimable only by TTL",
                        ipPort, orphanRequests, decodeEp.sampleInflightRequestIds(10));
            }
        }
        // Also surface scheduler-level inflight entries routed to this endpoint —
        // they can no longer complete via this worker's status reports.
        if (schedulerRole != null) {
            try {
                batchScheduler().logOrphanInflightForEndpoint(schedulerRole, ipPort);
            } catch (Exception e) {
                logger.debug("Scheduler orphan check skipped for {}: {}", ipPort, e.getMessage());
            }
        }
    }

    private FlexlbBatchScheduler batchScheduler() {
        return batchSchedulerFactory.getObject();
    }

    private PrefillEndpoint createPrefillEndpoint(WorkerStatus status, RoleType roleType,
                                                  String ipPort) {
        FlexlbConfig config = configService.loadBalanceConfig();
        prepareEndpointMetrics(roleType, status, ipPort);
        return new PrefillEndpoint(status, config, batchScheduler(), reporter);
    }

    private DecodeEndpoint createDecodeEndpoint(WorkerStatus status, String ipPort) {
        prepareEndpointMetrics(RoleType.DECODE, status, ipPort);
        return new DecodeEndpoint(status);
    }

    private SimpleWorkerEndpoint createSimpleEndpoint(WorkerStatus status, RoleType roleType,
                                                      String ipPort) {
        prepareEndpointMetrics(roleType, status, ipPort);
        return new SimpleWorkerEndpoint(status);
    }

    private void prepareEndpointMetrics(RoleType roleType, WorkerStatus status, String ipPort) {
        reporter.prepareEndpointMetrics(roleType.name(), status.getIp(), ipPort);
    }

    public void close() {
        prefillEndpoints.values().forEach(WorkerEndpoint::close);
        decodeEndpoints.values().forEach(WorkerEndpoint::close);
        pdFusionEndpoints.values().forEach(WorkerEndpoint::close);
        vitEndpoints.values().forEach(WorkerEndpoint::close);
    }

    /**
     * Expose all prefill endpoints for per-worker metrics reporting.
     */
    public ConcurrentHashMap<String, PrefillEndpoint> getPrefillEndpoints() {
        return prefillEndpoints;
    }

    /**
     * Expose all decode endpoints for per-worker metrics reporting.
     */
    public ConcurrentHashMap<String, DecodeEndpoint> getDecodeEndpoints() {
        return decodeEndpoints;
    }

    public int getEndpointCount(RoleType roleType) {
        return getEndpoints(roleType).size();
    }

    /**
     * Periodic TTL eviction for all endpoints.
     * <p>Each endpoint is responsible for its own inflight lifecycle.
     * This scheduled method provides a safety-net fallback for entries
     * that were not cleaned up by {@code calibrate()} (e.g., engine crash,
     * network partition, status report delay).
     * <p>Aggregates eviction counts by role and reports them via
     * {@code app.flexlb.inflight.ttl.expired.qps} tagged with PREFILL / DECODE.
     */
    @Scheduled(fixedRate = 60000L)
    public void scheduledEviction() {
        long ttlMs = configService.loadBalanceConfig().getFlexlbInflightTtlMs();
        int prefillExpired = 0;
        for (PrefillEndpoint ep : prefillEndpoints.values()) {
            prefillExpired += ep.evictExpiredBatches(ttlMs);
        }
        for (PrefillEndpoint ep : pdFusionEndpoints.values()) {
            prefillExpired += ep.evictExpiredBatches(ttlMs);
        }
        int decodeExpired = 0;
        for (DecodeEndpoint ep : decodeEndpoints.values()) {
            decodeExpired += ep.evictExpiredRequests(ttlMs);
        }
        if (prefillExpired > 0) {
            reporter.reportInflightTtlExpired(RoleType.PREFILL.name(), prefillExpired);
        }
        if (decodeExpired > 0) {
            reporter.reportInflightTtlExpired(RoleType.DECODE.name(), decodeExpired);
        }
    }
}
