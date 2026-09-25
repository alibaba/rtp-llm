package org.flexlb.engine.grpc.client;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EngineGrpcClientRetryPolicyTest {

    @Test
    void enqueueBatchDoesNotRetryAmbiguousBrokenConnection() {
        assertFalse(EngineGrpcClient.retriesBrokenConnections(
                EngineGrpcClient.ServiceType.BATCH_ENQUEUE));
    }

    @Test
    void retryPolicyChangeDoesNotDisableReadOnlyRpcRetries() {
        assertTrue(EngineGrpcClient.retriesBrokenConnections(
                EngineGrpcClient.ServiceType.WORKER_STATUS));
        assertTrue(EngineGrpcClient.retriesBrokenConnections(
                EngineGrpcClient.ServiceType.CACHE_STATUS));
        assertTrue(EngineGrpcClient.retriesBrokenConnections(
                EngineGrpcClient.ServiceType.MULTIMODAL_WORKER_STATUS));
        assertTrue(EngineGrpcClient.retriesBrokenConnections(
                EngineGrpcClient.ServiceType.MULTIMODAL_CACHE_STATUS));
    }

    @Test
    void cancelKeepsItsExistingSingleShotPolicy() {
        assertFalse(EngineGrpcClient.retriesBrokenConnections(
                EngineGrpcClient.ServiceType.ENGINE_CANCEL));
    }
}
