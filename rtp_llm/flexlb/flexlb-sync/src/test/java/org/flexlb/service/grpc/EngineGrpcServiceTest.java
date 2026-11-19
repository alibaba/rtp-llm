package org.flexlb.service.grpc;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.sync.runner.RunnerTestSupport;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EngineGrpcServiceTest {

    @ParameterizedTest
    @CsvSource({"DECODE,false", "PREFILL,true", "PDFUSION,true"})
    void cacheRequestOnlyFetchesDetailedKeysWhenNeeded(RoleType role, boolean needsCacheKeys) {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        EngineGrpcService service = new EngineGrpcService(client);
        WorkerStatus status = RunnerTestSupport.discovered(
                role, null, "127.0.0.1", 8080, 8081, "test-site");

        service.getCacheStatusAsync("127.0.0.1", 8081, status, 7L, 100L, role);

        ArgumentCaptor<EngineRpcService.CacheVersionPB> request =
                ArgumentCaptor.forClass(EngineRpcService.CacheVersionPB.class);
        verify(client).getCacheStatusAsync(eq("127.0.0.1"), eq(8081), request.capture(), eq(100L));
        assertEquals(needsCacheKeys, request.getValue().getNeedCacheKeys());
        assertEquals(7, request.getValue().getLatestCacheVersion());
    }
    @Test
    void firstControlledPollEnablesLeaseAndSubsequentPollCarriesRevocation() {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        ConfigService configs = mock(ConfigService.class);
        FlexlbConfig config = new FlexlbConfig();
        var lease = config.getRouter().getRoles().getPrefill().getChunkLease();
        lease.setEnabled(true);
        lease.setGroups(List.of("test"));
        when(configs.loadBalanceConfig()).thenReturn(config);
        EngineGrpcService service = new EngineGrpcService(client, configs);
        WorkerStatus worker = RunnerTestSupport.discovered(RoleType.PREFILL, "test", "127.0.0.1", 8080, 8081, "site");
        service.getWorkerStatusAsync("127.0.0.1", 8081, 7L, 100L, RoleType.PREFILL, worker);
        ArgumentCaptor<EngineRpcService.StatusVersionPB> request =
                ArgumentCaptor.forClass(EngineRpcService.StatusVersionPB.class);
        verify(client).getWorkerStatusAsync(eq("127.0.0.1"), eq(8081), request.capture(), eq(100L));
        assertTrue(request.getValue().hasPrefillChunkLease());
        assertTrue(request.getValue().getPrefillChunkLease().getEnabled());
        long firstVersion = request.getValue().getPrefillChunkLease().getVersion();
        worker.lock.lock();
        try {
            worker.prefillChunkLease().change(false);
        } finally {
            worker.lock.unlock();
        }
        org.mockito.Mockito.clearInvocations(client);
        service.getWorkerStatusAsync("127.0.0.1", 8081, 7L, 100L, RoleType.PREFILL, worker);
        verify(client).getWorkerStatusAsync(eq("127.0.0.1"), eq(8081), request.capture(), eq(100L));
        assertFalse(request.getValue().getPrefillChunkLease().getEnabled());
        assertTrue(request.getValue().getPrefillChunkLease().getVersion() > firstVersion);
        worker.lock.lock();
        try {
            worker.prefillChunkLease().confirm(request.getValue().getPrefillChunkLease().getVersion(), false);
        } finally {
            worker.lock.unlock();
        }
        org.mockito.Mockito.clearInvocations(client);
        service.getWorkerStatusAsync("127.0.0.1", 8081, 7L, 100L, RoleType.PREFILL, worker);
        verify(client).getWorkerStatusAsync(eq("127.0.0.1"), eq(8081), request.capture(), eq(100L));
        assertFalse(request.getValue().hasPrefillChunkLease(), "confirmed leases use ordinary status reads");
    }
}
