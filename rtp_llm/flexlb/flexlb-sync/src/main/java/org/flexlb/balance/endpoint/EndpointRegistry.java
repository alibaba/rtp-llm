package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.ObjectFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.LongPredicate;

@Component
public class EndpointRegistry {

    private final ConcurrentHashMap<String, PrefillEndpoint> prefillEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, PrefillEndpoint> pdFusionEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, SimpleWorkerEndpoint> vitEndpoints = new ConcurrentHashMap<>();
    private final ConfigService configService;
    private final ObjectFactory<FlexlbBatchScheduler> batchSchedulerFactory;
    private final BatchSchedulerReporter reporter;

    /** Hard age cap overriding TTL exemptions and observation keep-alives: 30 minutes. */
    private static final long INFLIGHT_HARD_MAX_AGE_MS = 30 * 60 * 1000L;
    /** Progress-aware batch-level inflight age cap: 120 seconds. */
    private static final long BATCH_INFLIGHT_MAX_AGE_MS = 120_000L;
    /** No-progress staleness threshold for the batch-level age cap: 60 seconds. */
    private static final long BATCH_INFLIGHT_STALE_MS = 60_000L;

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
        return batchSchedulerFactory.getObject();
    }

    private PrefillEndpoint createPrefillEndpoint(WorkerStatus status, RoleType roleType) {
        FlexlbConfig config = configService.loadBalanceConfig();
        prepareEndpointMetrics(roleType, status);
        return new PrefillEndpoint(status, config, batchScheduler(), reporter);
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
     * Trigger TTL eviction on all prefill and decode endpoints.
     *
     * @param ttlMs        max age before eviction
     * @param hardMaxAgeMs hard age cap overriding TTL exemptions and
     *                     observation keep-alives; {@code <= 0} disables it
     * @param batchInflightMaxAgeMs progress-aware batch-level inflight age
     *                     cap applied to prefill/pdFusion batch
     *                     ledgers — force-settles over-age batches that also
     *                     went unobserved; {@code <= 0} disables it
     * @param batchInflightStaleMs  no-progress staleness threshold for the
     *                     age cap; {@code <= 0} drops the progress guard
     */
    private void evictExpiredAll(long ttlMs, long hardMaxAgeMs,
                                 long batchInflightMaxAgeMs, long batchInflightStaleMs) {
        // Race guard for hard-cap eviction: entries the scheduler still owns
        // are left to the scheduler's own cleanup cascade. The batch age
        // cap is intentionally unconditional (bounded freeze) and takes the
        // scheduler-owned members through the handler terminal chain instead.
        LongPredicate schedulerOwns = batchScheduler()::hasInflightRequest;
        prefillEndpoints.forEach((endpoint, ep) ->
                reportEndpointEviction(RoleType.PREFILL, endpoint, ep.getIp(),
                        ep.evictExpiredBatchesByReason(ttlMs, hardMaxAgeMs, batchInflightMaxAgeMs,
                                batchInflightStaleMs, schedulerOwns), ttlMs));
        decodeEndpoints.forEach((endpoint, ep) ->
                reportEndpointEviction(RoleType.DECODE, endpoint, ep.getIp(),
                        ep.evictExpiredRequestsByReason(ttlMs, hardMaxAgeMs, schedulerOwns), ttlMs));
        pdFusionEndpoints.forEach((endpoint, ep) ->
                reportEndpointEviction(RoleType.PDFUSION, endpoint, ep.getIp(),
                        ep.evictExpiredBatchesByReason(ttlMs, hardMaxAgeMs, batchInflightMaxAgeMs,
                                batchInflightStaleMs, schedulerOwns), ttlMs));
    }

    /**
     * Report one endpoint's eviction sweep on the shared
     * {@code app.flexlb.inflight.ttl.expired.qps} series, split by exit
     * reason. Reason buckets mirror the eviction exits and reuse the
     * scheduler-side series naming: {@code all_terminal} — every member's
     * scheduler-side future is already terminal (all-terminal release);
     * {@code age_capped} — progress-aware batch-level age cap;
     * {@code hard_age_cap} — guarded hard cap overriding fences and
     * observation keep-alives; {@code ttl} — normal unobserved TTL. Only
     * non-zero buckets are reported, so decode endpoints (no batch-ledger
     * exits) never emit {@code all_terminal}/{@code age_capped}.
     */
    private void reportEndpointEviction(RoleType role,
                                        String endpoint,
                                        String engineIp,
                                        EvictionBreakdown evictions,
                                        long ttlMs) {
        int evicted = evictions.total();
        if (evicted > 0) {
            Logger.info("event=endpoint_inflight_ttl_eviction role={} endpoint={} "
                            + "evicted={} ttl_ms={} all_terminal={} age_capped={} "
                            + "hard_age_cap={} ttl={}",
                    role, endpoint, evicted, ttlMs, evictions.allTerminal(),
                    evictions.ageCapped(), evictions.hardAgeCap(), evictions.ttl());
            reportEvictionReason(role, engineIp, "all_terminal", evictions.allTerminal());
            reportEvictionReason(role, engineIp, "age_capped", evictions.ageCapped());
            reportEvictionReason(role, engineIp, "hard_age_cap", evictions.hardAgeCap());
            reportEvictionReason(role, engineIp, "ttl", evictions.ttl());
        }
    }

    private void reportEvictionReason(RoleType role, String engineIp, String reason, int count) {
        if (count > 0) {
            reporter.reportEndpointInflightTtlExpired(role.name(), engineIp, reason, count);
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
        FlexlbConfig config = configService.loadBalanceConfig();
        evictExpiredAll(config.getFlexlbInflightTtlMs(),
                INFLIGHT_HARD_MAX_AGE_MS,
                BATCH_INFLIGHT_MAX_AGE_MS,
                BATCH_INFLIGHT_STALE_MS);
    }
}
