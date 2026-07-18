package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.MetricLease;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.springframework.beans.factory.ObjectFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

@Component
public class EndpointRegistry {

    private static final long MIN_ENDPOINT_METRIC_RETIREMENT_GRACE_MS = 10_000L;

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

    public WorkerEndpoint get(String ipPort) {
        WorkerEndpoint ep = prefillEndpoints.get(ipPort);
        if (ep != null) {
            return ep;
        }
        ep = decodeEndpoints.get(ipPort);
        if (ep != null) {
            return ep;
        }
        ep = pdFusionEndpoints.get(ipPort);
        if (ep != null) {
            return ep;
        }
        return vitEndpoints.get(ipPort);
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

    public PrefillEndpoint getPdFusion(String ipPort) {
        return pdFusionEndpoints.get(ipPort);
    }

    public SimpleWorkerEndpoint getVit(String ipPort) {
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

    public PrefillEndpoint ensurePrefillEndpoint(String ipPort, WorkerStatus status) {
        return ensurePrefillEndpoint(ipPort, status, RoleType.PREFILL);
    }

    private PrefillEndpoint ensurePrefillEndpoint(String ipPort, WorkerStatus status, RoleType roleType) {
        PrefillEndpoint endpoint = prefillEndpoints.get(ipPort);
        if (endpoint != null && endpoint.getStatus() == status) {
            return endpoint;
        }
        return ensureEndpoint(prefillEndpoints, ipPort, status,
                candidateStatus -> createPrefillEndpoint(candidateStatus, roleType, ipPort));
    }

    public DecodeEndpoint ensureDecodeEndpoint(String ipPort, WorkerStatus status) {
        return ensureEndpoint(decodeEndpoints, ipPort, status,
                candidateStatus -> createDecodeEndpoint(candidateStatus, ipPort));
    }

    public PrefillEndpoint ensurePdFusionEndpoint(String ipPort, WorkerStatus status) {
        return ensurePdFusionEndpoint(ipPort, status, RoleType.PDFUSION);
    }

    private PrefillEndpoint ensurePdFusionEndpoint(String ipPort, WorkerStatus status, RoleType roleType) {
        PrefillEndpoint endpoint = pdFusionEndpoints.get(ipPort);
        if (endpoint != null && endpoint.getStatus() == status) {
            return endpoint;
        }
        return ensureEndpoint(pdFusionEndpoints, ipPort, status,
                candidateStatus -> createPrefillEndpoint(candidateStatus, roleType, ipPort));
    }

    public SimpleWorkerEndpoint ensureVitEndpoint(String ipPort, WorkerStatus status) {
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
        endpoint.close();
        return true;
    }

    private FlexlbBatchScheduler batchScheduler() {
        return batchSchedulerFactory == null ? null : batchSchedulerFactory.getObject();
    }

    private MetricLease acquireEndpointMetrics(RoleType roleType, WorkerStatus status, String ipPort) {
        if (reporter == null || roleType == null || status == null) {
            return MetricLease.noop();
        }
        return acquireEndpointMetrics(
                roleType, status, ipPort, configService.loadBalanceConfig());
    }

    private MetricLease acquireEndpointMetrics(RoleType roleType, WorkerStatus status, String ipPort,
                                               FlexlbConfig config) {
        if (reporter == null || roleType == null || status == null) {
            return MetricLease.noop();
        }
        long enqueueDeadlineMs = Math.max(0L, config.getFlexlbBatchEnqueueDeadlineMs());
        long retirementGraceMs = Math.max(
                MIN_ENDPOINT_METRIC_RETIREMENT_GRACE_MS,
                enqueueDeadlineMs >= Long.MAX_VALUE / 2L
                        ? Long.MAX_VALUE
                        : enqueueDeadlineMs * 2L);
        MetricLease lease = reporter.acquireEndpointMetrics(
                roleType.name(), status.getIp(), ipPort, retirementGraceMs);
        return lease == null ? MetricLease.noop() : lease;
    }

    private PrefillEndpoint createPrefillEndpoint(WorkerStatus status, RoleType roleType,
                                                  String ipPort) {
        FlexlbConfig config = configService.loadBalanceConfig();
        MetricLease lease = acquireEndpointMetrics(roleType, status, ipPort, config);
        try {
            return new PrefillEndpoint(status, config, batchScheduler(), reporter, lease);
        } catch (RuntimeException | Error e) {
            lease.close();
            throw e;
        }
    }

    private DecodeEndpoint createDecodeEndpoint(WorkerStatus status, String ipPort) {
        MetricLease lease = acquireEndpointMetrics(RoleType.DECODE, status, ipPort);
        try {
            return new DecodeEndpoint(status, lease);
        } catch (RuntimeException | Error e) {
            lease.close();
            throw e;
        }
    }

    private SimpleWorkerEndpoint createSimpleEndpoint(WorkerStatus status, RoleType roleType,
                                                      String ipPort) {
        MetricLease lease = acquireEndpointMetrics(roleType, status, ipPort);
        try {
            return new SimpleWorkerEndpoint(status, lease);
        } catch (RuntimeException | Error e) {
            lease.close();
            throw e;
        }
    }

    /**
     * Replace prefill endpoint at given key. Closes old endpoint if present.
     * Note: This is primarily used in tests. Production code should use ensurePrefillEndpoint().
     */
    public void putPrefill(String ipPort, PrefillEndpoint endpoint) {
        PrefillEndpoint old = prefillEndpoints.put(ipPort, endpoint);
        if (old != null && old != endpoint) {
            old.close();
        }
    }

    /**
     * Replace decode endpoint at given key. Closes old endpoint if present.
     * Note: This is primarily used in tests. Production code should use ensureDecodeEndpoint().
     */
    public void putDecode(String ipPort, DecodeEndpoint endpoint) {
        DecodeEndpoint old = decodeEndpoints.put(ipPort, endpoint);
        if (old != null && old != endpoint) {
            old.close();
        }
    }

    public void putPdFusion(String ipPort, PrefillEndpoint endpoint) {
        PrefillEndpoint old = pdFusionEndpoints.put(ipPort, endpoint);
        if (old != null && old != endpoint) {
            old.close();
        }
    }

    public void putVit(String ipPort, SimpleWorkerEndpoint endpoint) {
        putSimple(vitEndpoints, ipPort, endpoint);
    }

    private void putSimple(ConcurrentHashMap<String, SimpleWorkerEndpoint> endpoints,
                           String ipPort, SimpleWorkerEndpoint endpoint) {
        SimpleWorkerEndpoint old = endpoints.put(ipPort, endpoint);
        if (old != null && old != endpoint) {
            old.close();
        }
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

    public ConcurrentHashMap<String, PrefillEndpoint> getPdFusionEndpoints() {
        return pdFusionEndpoints;
    }

    public ConcurrentHashMap<String, SimpleWorkerEndpoint> getVitEndpoints() {
        return vitEndpoints;
    }

    public int getEndpointCount(RoleType roleType) {
        return getEndpoints(roleType).size();
    }

    /**
     * Trigger TTL eviction on all prefill and decode endpoints.
     *
     * @param ttlMs max age before eviction
     */
    public void evictExpiredAll(long ttlMs) {
        prefillEndpoints.values().forEach(ep -> ep.evictExpiredBatches(ttlMs));
        decodeEndpoints.values().forEach(ep -> ep.evictExpiredRequests(ttlMs));
        pdFusionEndpoints.values().forEach(ep -> ep.evictExpiredBatches(ttlMs));
    }

    /**
     * Periodic TTL eviction for all endpoints.
     * <p>Each endpoint is responsible for its own inflight lifecycle.
     * This scheduled method provides a safety-net fallback for entries
     * that were not cleaned up by {@code calibrate()} (e.g., engine crash,
     * network partition, status report delay).
     */
    @Scheduled(fixedRate = 60000L)
    public void scheduledEviction() {
        long ttlMs = configService.loadBalanceConfig().getFlexlbInflightTtlMs();
        evictExpiredAll(ttlMs);
    }
}
