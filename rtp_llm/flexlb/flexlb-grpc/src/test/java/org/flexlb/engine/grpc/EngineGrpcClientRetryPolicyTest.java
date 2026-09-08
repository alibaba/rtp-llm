package org.flexlb.engine.grpc;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EngineGrpcClientRetryPolicyTest {

    @Test
    void enqueueBatchDoesNotRetryAmbiguousBrokenConnection() {
        assertFalse(EngineGrpcClient.retriesBrokenConnections(
                AbstractGrpcClient.ServiceType.BATCH_ENQUEUE));
    }

    @Test
    void retryPolicyChangeDoesNotDisableReadOnlyRpcRetries() {
        assertTrue(EngineGrpcClient.retriesBrokenConnections(
                AbstractGrpcClient.ServiceType.WORKER_STATUS));
        assertTrue(EngineGrpcClient.retriesBrokenConnections(
                AbstractGrpcClient.ServiceType.CACHE_STATUS));
        assertTrue(EngineGrpcClient.retriesBrokenConnections(
                AbstractGrpcClient.ServiceType.MULTIMODAL_WORKER_STATUS));
        assertTrue(EngineGrpcClient.retriesBrokenConnections(
                AbstractGrpcClient.ServiceType.MULTIMODAL_CACHE_STATUS));
    }

    @Test
    void cancelKeepsItsExistingSingleShotPolicy() {
        assertFalse(EngineGrpcClient.retriesBrokenConnections(
                AbstractGrpcClient.ServiceType.ENGINE_CANCEL));
    }
}
