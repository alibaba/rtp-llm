package org.flexlb.cache.match.kvcm;

import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.flexlb.exception.KvcmQueryException;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class KvcmCacheMatchProviderTest {

    @Test
    void delegatesCacheMatchingToKvcmClient() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        when(client.findMatchingEngines(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default"))
                .thenReturn(Mono.just(Map.of("10.0.0.1:8080", 1)));
        KvcmCacheMatchProvider provider = new KvcmCacheMatchProvider(client);

        StepVerifier.create(provider.findMatchingEngines(
                        "request-1", List.of(11L), 2192, RoleType.PREFILL, "default"))
                .expectNext(Map.of("10.0.0.1:8080", 1))
                .verifyComplete();
        verify(client).findMatchingEngines(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default");
    }

    @Test
    void propagatesClientFailureWithoutAdditionalProviderRetry() {
        KvcmGrpcClient client = mock(KvcmGrpcClient.class);
        KvcmQueryException failure = new KvcmQueryException("unavailable");
        when(client.findMatchingEngines(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default"))
                .thenReturn(Mono.error(failure));
        KvcmCacheMatchProvider provider = new KvcmCacheMatchProvider(client);

        StepVerifier.create(provider.findMatchingEngines(
                        "request-1", List.of(11L), 2192, RoleType.PREFILL, "default"))
                .expectErrorSatisfies(error -> assertSame(failure, error))
                .verify();

        verify(client).findMatchingEngines(
                "request-1", List.of(11L), 2192, RoleType.PREFILL, "default");
    }
}
