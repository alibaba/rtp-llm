package org.flexlb.sync.status;

import org.flexlb.cache.domain.BlockHashConfig;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Collection;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class WorkerBlockHashConfigResolverTest {

    private final Map<RoleType, Map<String, WorkerStatus>> workerStatuses =
            new EnumMap<>(RoleType.class);
    private final WorkerStatusProvider workerStatusProvider = this::getWorkerStatuses;
    private WorkerBlockHashConfigResolver resolver;
    private int nextPort = 8000;

    @BeforeEach
    void setUp() {
        clearWorkerStatuses();
        resolver = new WorkerBlockHashConfigResolver(workerStatusProvider);
    }

    @AfterEach
    void tearDown() {
        resolver.shutdown();
        clearWorkerStatuses();
    }

    @Test
    void resolvesAndCachesConfigFromHealthyPrefillWorker() {
        statusMap(RoleType.PREFILL).put("10.0.0.1:8000",
                worker(RoleType.PREFILL, 64, 1));

        assertEquals(config(64, 1), resolver.resolve());

        statusMap(RoleType.PREFILL).clear();
        resolver.refresh();
        assertEquals(config(64, 1), resolver.resolve());
    }

    @Test
    void refreshesConfigWhenPrefillWorkersChangeConsistently() {
        statusMap(RoleType.PREFILL).put("10.0.0.1:8000",
                worker(RoleType.PREFILL, 64, 1));
        resolver.refresh();

        statusMap(RoleType.PREFILL).clear();
        statusMap(RoleType.PREFILL).put("10.0.0.2:8000",
                worker(RoleType.PREFILL, 16, 0));
        resolver.refresh();

        assertEquals(config(16, 0), resolver.resolve());
    }

    @Test
    void keepsLastValidConfigWhenPrefillWorkersAreInconsistent() {
        statusMap(RoleType.PREFILL).put("10.0.0.1:8000",
                worker(RoleType.PREFILL, 64, 1));
        resolver.refresh();
        statusMap(RoleType.PREFILL).put("10.0.0.2:8000",
                worker(RoleType.PREFILL, 64, 0));
        resolver.refresh();

        assertEquals(config(64, 1), resolver.resolve());
    }

    @Test
    void ignoresPdFusionConfigWhenPrefillConfigIsAvailable() {
        statusMap(RoleType.PREFILL).put("10.0.0.1:8000",
                worker(RoleType.PREFILL, 64, 1));
        statusMap(RoleType.PDFUSION).put("10.0.0.2:8000",
                worker(RoleType.PDFUSION, 64, 0));

        assertEquals(config(64, 1), resolver.resolve());
    }

    @Test
    void fallsBackToPdFusionWhenNoPrefillConfigIsAvailable() {
        statusMap(RoleType.PDFUSION).put("10.0.0.2:8000",
                worker(RoleType.PDFUSION, 64, 0));

        assertEquals(config(64, 0), resolver.resolve());
    }

    @Test
    void fallsBackToPdFusionWhenPrefillWorkersHaveNoUsableConfig() {
        statusMap(RoleType.PREFILL).put("10.0.0.1:8000",
                worker(RoleType.PREFILL, 64, 1, false));
        statusMap(RoleType.PREFILL).put("10.0.0.2:8000",
                worker(RoleType.PREFILL, 0, 1));
        statusMap(RoleType.PDFUSION).put("10.0.0.3:8000",
                worker(RoleType.PDFUSION, 64, 0));

        assertEquals(config(64, 0), resolver.resolve());
    }

    @Test
    void failsWhenNoHealthyWorkerProvidesConfig() {
        assertThrows(IllegalStateException.class, resolver::resolve);
    }

    private Collection<WorkerStatus> getWorkerStatuses(RoleType roleType, String group) {
        Map<String, WorkerStatus> statuses = workerStatuses.get(roleType);
        return statuses == null ? List.of() : statuses.values();
    }

    private void clearWorkerStatuses() {
        for (RoleType roleType : RoleType.values()) {
            statusMap(roleType).clear();
        }
    }

    private BlockHashConfig config(long blockSize, int lookaheadTokens) {
        return new BlockHashConfig(blockSize, lookaheadTokens);
    }

    private Map<String, WorkerStatus> statusMap(RoleType roleType) {
        return workerStatuses.computeIfAbsent(roleType, ignored -> new HashMap<>());
    }

    private WorkerStatus worker(RoleType roleType, long blockSize, int lookaheadTokens) {
        return worker(roleType, blockSize, lookaheadTokens, true);
    }

    private WorkerStatus worker(
            RoleType roleType, long blockSize, int lookaheadTokens, boolean alive) {
        int port = nextPort++;
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setBlockSize(blockSize);
        WorkerStatus workerStatus = WorkerStatus.createDiscovered(
                roleType, null, "127.0.0.1", port, port + 1, "test-site");
        workerStatus.publishCacheStatus(cacheStatus);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(roleType);
        response.setAlive(alive);
        response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());
        response.setBlockHashLookaheadTokens(lookaheadTokens);
        workerStatus.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = workerStatus.prepareNewStatus(
                    workerStatus.freezeStatusResponse(response));
            workerStatus.publishPreparedStatus(prepared);
            workerStatus.recordSuccessfulPoll(alive);
        } finally {
            workerStatus.lock.unlock();
        }
        return workerStatus;
    }
}
