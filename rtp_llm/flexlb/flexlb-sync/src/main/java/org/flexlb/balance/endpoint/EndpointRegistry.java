package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchDecisionHandler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.ObjectFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Collections;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

@Component
public class EndpointRegistry {

    private final ConcurrentHashMap<String, PrefillEndpoint> prefillEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, PrefillEndpoint> pdFusionEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, SimpleWorkerEndpoint> vitEndpoints = new ConcurrentHashMap<>();
    private final Map<String, PrefillEndpoint> prefillEndpointView =
            Collections.unmodifiableMap(prefillEndpoints);
    private final Map<String, DecodeEndpoint> decodeEndpointView =
            Collections.unmodifiableMap(decodeEndpoints);
    private final Map<String, PrefillEndpoint> pdFusionEndpointView =
            Collections.unmodifiableMap(pdFusionEndpoints);
    private final Map<String, SimpleWorkerEndpoint> vitEndpointView =
            Collections.unmodifiableMap(vitEndpoints);
    private final ConfigService configService;
    private final ObjectFactory<BatchDecisionHandler> batchDecisionHandlerFactory;
    private final BatchSchedulerReporter reporter;

    public EndpointRegistry(ConfigService configService,
                            ObjectFactory<BatchDecisionHandler> batchDecisionHandlerFactory,
                            BatchSchedulerReporter reporter) {
        this.configService = configService;
        this.batchDecisionHandlerFactory = batchDecisionHandlerFactory;
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
            return prefillEndpointView;
        }
        if (roleType == RoleType.DECODE) {
            return decodeEndpointView;
        }
        if (roleType == RoleType.PDFUSION) {
            return pdFusionEndpointView;
        }
        if (roleType == RoleType.VIT) {
            return vitEndpointView;
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
        if (!supportsTopology(roleType, status)) {
            throw new UnsupportedOperationException(
                    roleType + " endpoint does not support dp_size=" + status.getDpSize());
        }
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

    /**
     * Apply one current-generation worker response to the endpoint domain.
     * Scheduler lifecycle processing is intentionally not part of this API.
     */
    public void updateEndpointFromWorkerStatus(
            WorkerStatus status, WorkerStatusResponse response) {
        RoleType role = validateStatusResponse(status, response);
        String ipPort = status.getIpPort();
        if (!isEndpointRole(role)) {
            status.setReportedSchedulingLoad(0);
            return;
        }
        if (!status.isAlive()) {
            remove(role, ipPort, status);
            status.setReportedSchedulingLoad(0);
            return;
        }
        if (!supportsTopology(role, status)) {
            remove(role, ipPort, status);
            status.setReportedSchedulingLoad(0);
            return;
        }

        WorkerEndpoint previous = get(role, ipPort);
        boolean installsEndpoint = previous == null || previous.getStatus() != status;
        WorkerEndpoint endpoint = ensureEndpoint(role, ipPort, status);
        try {
            endpoint.applyWorkerStatusResponse(status, response);
        } catch (RuntimeException | Error applyFailure) {
            if (installsEndpoint) {
                try {
                    remove(role, ipPort, status);
                } catch (RuntimeException | Error cleanupFailure) {
                    applyFailure.addSuppressed(cleanupFailure);
                }
            }
            throw applyFailure;
        }
        status.setReportedSchedulingLoad(endpoint.schedulingLoad());
    }

    /**
     * Refresh existing endpoint activity anchors for an equal-version response.
     * Missing, retired, or topology-changing endpoints fall back to the full
     * update path; the common case performs no absence reconciliation.
     */
    public void refreshEndpointActivity(WorkerStatus status, WorkerStatusResponse response) {
        RoleType role = validateStatusResponse(status, response);
        WorkerEndpoint endpoint = get(role, status.getIpPort());
        if (!isEndpointRole(role) || !status.isAlive() || !supportsTopology(role, status)
                || endpoint == null || endpoint.getStatus() != status) {
            updateEndpointFromWorkerStatus(status, response);
            return;
        }
        endpoint.refreshWorkerStatusActivity(status, response);
        status.setReportedSchedulingLoad(endpoint.schedulingLoad());
    }

    private static RoleType validateStatusResponse(
            WorkerStatus status, WorkerStatusResponse response) {
        if (status == null || response == null || status.getRole() == null) {
            throw new IllegalArgumentException("status, response and role are required");
        }
        RoleType role = status.getRole();
        if (response.getRole() != role) {
            throw new IllegalArgumentException(
                    "WorkerStatus role mismatch: expected=" + role + ", response=" + response.getRole());
        }
        return role;
    }

    private static boolean isEndpointRole(RoleType role) {
        return role == RoleType.PREFILL || role == RoleType.DECODE
                || role == RoleType.PDFUSION || role == RoleType.VIT;
    }

    private static boolean supportsTopology(RoleType role, WorkerStatus status) {
        return (role != RoleType.PREFILL && role != RoleType.PDFUSION)
                || status.getDpSize() <= 1;
    }

    private PrefillEndpoint ensurePrefillEndpoint(String ipPort, WorkerStatus status, RoleType roleType) {
        PrefillEndpoint endpoint = prefillEndpoints.get(ipPort);
        if (endpoint != null && endpoint.getStatus() == status) {
            return endpoint;
        }
        return ensureEndpoint(prefillEndpoints, ipPort, status,
                candidateStatus -> createPrefillEndpoint(candidateStatus, roleType));
    }

    private DecodeEndpoint ensureDecodeEndpoint(String ipPort, WorkerStatus status) {
        return ensureEndpoint(decodeEndpoints, ipPort, status,
                this::createDecodeEndpoint);
    }

    private PrefillEndpoint ensurePdFusionEndpoint(String ipPort, WorkerStatus status, RoleType roleType) {
        PrefillEndpoint endpoint = pdFusionEndpoints.get(ipPort);
        if (endpoint != null && endpoint.getStatus() == status) {
            return endpoint;
        }
        return ensureEndpoint(pdFusionEndpoints, ipPort, status,
                candidateStatus -> createPrefillEndpoint(candidateStatus, roleType));
    }

    private SimpleWorkerEndpoint ensureVitEndpoint(String ipPort, WorkerStatus status) {
        return ensureEndpoint(vitEndpoints, ipPort, status,
                candidateStatus -> createSimpleEndpoint(candidateStatus, RoleType.VIT));
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
            current.beginRetirement();
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

    /**
     * Stop new generation-specific work before the WorkerStatus map changes.
     * The exact-status check prevents an old retirement from fencing a newer
     * endpoint generation at the same address.
     */
    public boolean beginEndpointRetirement(
            RoleType roleType, String ipPort, WorkerStatus expectedStatus) {
        WorkerEndpoint endpoint = get(roleType, ipPort);
        if (endpoint == null || endpoint.getStatus() != expectedStatus) {
            return false;
        }
        endpoint.beginRetirement();
        return true;
    }

    private <T extends WorkerEndpoint> boolean remove(ConcurrentHashMap<String, T> endpoints,
                                                       String ipPort,
                                                       WorkerStatus expectedStatus) {
        T endpoint = endpoints.get(ipPort);
        if (endpoint == null || endpoint.getStatus() != expectedStatus) {
            return false;
        }
        endpoint.beginRetirement();
        if (!endpoints.remove(ipPort, endpoint)) {
            return false;
        }
        expectedStatus.setReportedSchedulingLoad(0);
        endpoint.close();
        return true;
    }

    private BatchDecisionHandler batchDecisionHandler() {
        return batchDecisionHandlerFactory.getObject();
    }

    private PrefillEndpoint createPrefillEndpoint(WorkerStatus status, RoleType roleType) {
        FlexlbConfig config = configService.loadBalanceConfig();
        prepareEndpointMetrics(roleType, status);
        return new PrefillEndpoint(status, config, batchDecisionHandler(), reporter);
    }

    private DecodeEndpoint createDecodeEndpoint(WorkerStatus status) {
        prepareEndpointMetrics(RoleType.DECODE, status);
        return new DecodeEndpoint(status);
    }

    private SimpleWorkerEndpoint createSimpleEndpoint(WorkerStatus status, RoleType roleType) {
        prepareEndpointMetrics(roleType, status);
        return new SimpleWorkerEndpoint(status);
    }

    private void prepareEndpointMetrics(RoleType roleType, WorkerStatus status) {
        reporter.prepareEndpointMetrics(roleType.name(), status.getIp());
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
    public Map<String, PrefillEndpoint> getPrefillEndpoints() {
        return prefillEndpointView;
    }

    /**
     * Expose all decode endpoints for per-worker metrics reporting.
     */
    public Map<String, DecodeEndpoint> getDecodeEndpoints() {
        return decodeEndpointView;
    }

    public int getEndpointCount(RoleType roleType) {
        return getEndpoints(roleType).size();
    }

    /**
     * Trigger TTL eviction on all prefill and decode endpoints.
     *
     * @param ttlMs max age before eviction
     */
    private void evictExpiredAll(long ttlMs) {
        prefillEndpoints.forEach((endpoint, ep) ->
                logEndpointEviction(RoleType.PREFILL, endpoint,
                        ep.evictExpiredBatches(ttlMs), ttlMs));
        decodeEndpoints.forEach((endpoint, ep) ->
                logEndpointEviction(RoleType.DECODE, endpoint,
                        ep.evictExpiredRequests(ttlMs), ttlMs));
        pdFusionEndpoints.forEach((endpoint, ep) ->
                logEndpointEviction(RoleType.PDFUSION, endpoint,
                        ep.evictExpiredBatches(ttlMs), ttlMs));
    }

    private static void logEndpointEviction(RoleType role,
                                            String endpoint,
                                            int evicted,
                                            long ttlMs) {
        if (evicted > 0) {
            Logger.info("event=endpoint_inflight_ttl_eviction role={} endpoint={} "
                            + "evicted={} ttl_ms={}",
                    role, endpoint, evicted, ttlMs);
        }
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
