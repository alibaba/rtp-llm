package org.flexlb.cache.match.kvcm;

import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.flexlb.exception.KvcmQueryException;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class KvcmCacheMatchProviderTest {

    @Test
    void delegatesCacheMatchingToKvcmClient() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        when(client.findMatchingEngines(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default"))
                .thenReturn(Map.of("10.0.0.1:8080", new HostCacheMatch(1, 0, 1)));
        KvcmCacheMatchProvider provider = new KvcmCacheMatchProvider(client);

        Map<String, HostCacheMatch> result = provider.findMatchingEngines(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default");

        assertEquals(1, result.get("10.0.0.1:8080").localMatchBlocks());
        verify(client).findMatchingEngines(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default");
    }

    @Test
    void propagatesClientFailureWithoutAdditionalProviderRetry() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        KvcmQueryException failure = new KvcmQueryException("unavailable");
        when(client.findMatchingEngines(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default"))
                .thenThrow(failure);
        KvcmCacheMatchProvider provider = new KvcmCacheMatchProvider(client);

        KvcmQueryException thrown = assertThrows(KvcmQueryException.class,
                () -> provider.findMatchingEngines(
                        "request-1", List.of(11L), 2192, RoleType.PREFILL, "default"));
        assertSame(failure, thrown);

        verify(client).findMatchingEngines(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default");
    }
}
