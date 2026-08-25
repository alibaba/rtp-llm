package org.flexlb.it.fixture.kvcm;

import org.flexlb.dao.route.RoleType;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;

import java.io.IOException;
import java.util.List;

/**
 * Focused scripting facade for the KVCM boundary used by cache-oriented integration scenarios.
 *
 * <p>Engine worker status is intentionally owned by {@link IntegrationTestFixtures}; this facade
 * owns only KVCM discovery, cache-match responses, and wire-observation state.
 */
public final class KvcmIntegrationTestFixtures {

    private static final ScriptedKvcm KVCM = new ScriptedKvcm();

    private KvcmIntegrationTestFixtures() {}

    /** Responses used to model a valid miss, configured hit, or transport failure. */
    public enum CacheResponse {
        EMPTY,
        CONFIGURED_WORKER_MATCH,
        UNAVAILABLE
    }

    /** Starts the KVCM fake and returns its dynamically allocated gRPC port. */
    public static synchronized int startKvcm() {
        try {
            KVCM.start();
            return KVCM.port();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to start scripted KVCM", e);
        }
    }

    /** Selects the worker reported as the KVCM cache-match host. */
    public static void setMatchingWorker(RoleType roleType, int index) {
        KVCM.service().setMatchingWorkerHost(IntegrationTestFixtures.workerIpPort(roleType, index));
    }

    /** Sets the result or failure served by the scripted KVCM cache query. */
    public static void setCacheResponse(CacheResponse response) {
        KVCM.service().setCacheResponse(response);
    }

    /** Sets the configured worker's local-match block count, or {@code -1} for every request block. */
    public static void setLocalMatchBlocks(int localMatchBlocks) {
        KVCM.service().setLocalMatchBlocks(localMatchBlocks);
    }

    /** Returns cache-state query calls observed by the fake. */
    public static int cacheStateCalls() {
        return KVCM.service().cacheStateCalls();
    }

    /** Returns leader-discovery calls observed by the fake. */
    public static int clusterInfoCalls() {
        return KVCM.service().clusterInfoCalls();
    }

    /** Returns the exact block keys from the most recent KVCM request. */
    public static List<Long> lastCacheBlockKeys() {
        return KVCM.service().lastCacheBlockKeys();
    }
}
