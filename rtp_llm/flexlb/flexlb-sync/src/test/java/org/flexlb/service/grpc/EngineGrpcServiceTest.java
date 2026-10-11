package org.flexlb.service.grpc;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.sync.runner.RunnerTestSupport;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

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
}
