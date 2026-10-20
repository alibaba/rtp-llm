package org.flexlb.service.grpc;

import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

class WorkerStatusRpcClientTest {

    @org.junit.jupiter.api.Test
    void cacheVersionPreservesItsFullInt64Range() {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        long version = (long) Integer.MAX_VALUE + 10L;
        new WorkerStatusRpcClient(client).getCacheStatusAsync("127.0.0.1", 8081, version, 100L, RoleType.PREFILL);
        var request = ArgumentCaptor.forClass(EngineRpcService.CacheVersionPB.class);
        verify(client).getCacheStatusAsync(eq("127.0.0.1"), eq(8081), request.capture(), eq(100L));
        assertEquals(version, request.getValue().getLatestCacheVersion());
    }

    @ParameterizedTest
    @CsvSource({"DECODE,false", "PREFILL,true", "PDFUSION,true", "VIT,false"})
    void cacheRequestOnlyFetchesDetailedKeysWhenNeeded(RoleType role, boolean needsCacheKeys) {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        WorkerStatusRpcClient service = new WorkerStatusRpcClient(client);

        service.getCacheStatusAsync("127.0.0.1", 8081, 7L, 100L, role);

        ArgumentCaptor<EngineRpcService.CacheVersionPB> request =
                ArgumentCaptor.forClass(EngineRpcService.CacheVersionPB.class);
        if (role == RoleType.VIT) {
            verify(client).getMultimodalCacheStatusAsync(eq("127.0.0.1"), eq(8081), request.capture(), eq(100L));
        } else {
            verify(client).getCacheStatusAsync(eq("127.0.0.1"), eq(8081), request.capture(), eq(100L));
        }
        verifyNoMoreInteractions(client);
        assertEquals(needsCacheKeys, request.getValue().getNeedCacheKeys());
        assertEquals(7, request.getValue().getLatestCacheVersion());
    }
}
