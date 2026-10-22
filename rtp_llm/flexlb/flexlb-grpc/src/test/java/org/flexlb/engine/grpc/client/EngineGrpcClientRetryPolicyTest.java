package org.flexlb.engine.grpc.client;

import com.google.protobuf.InvalidProtocolBufferException;
import io.grpc.Status;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.nio.channels.ClosedChannelException;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class EngineGrpcClientRetryPolicyTest {

    @ParameterizedTest
    @EnumSource(
            value = GrpcFailure.Kind.class,
            names = {"LOCAL_MEMORY_EXHAUSTED", "LOCAL_EXECUTOR_REJECTED", "LOCAL_BIND_FAILED"})
    void originalLocalSocketFailurePreventsRetryOfTheResultingChannelClose(GrpcFailure.Kind cause) {
        GrpcCallTrace trace = mock(GrpcCallTrace.class);
        when(trace.snapshot())
                .thenReturn(
                        new GrpcCallTrace.Snapshot(
                                GrpcCallTrace.Stage.WAITING_RESPONSE,
                                "READ_FAILURE",
                                cause,
                                1,
                                -1,
                                0,
                                1,
                                -1,
                                -1,
                                10,
                                11,
                                9));

        assertFalse(
                EngineGrpcClient.shouldRetry(
                        AbstractGrpcClient.ServiceType.WORKER_STATUS,
                        Status.UNAVAILABLE
                                .withCause(new ClosedChannelException())
                                .asRuntimeException(),
                        trace));
    }

    @Test
    void peerEofStillAllowsTheExistingReadOnlyRetry() {
        assertTrue(
                EngineGrpcClient.shouldRetry(
                        AbstractGrpcClient.ServiceType.WORKER_STATUS,
                        Status.UNAVAILABLE.asRuntimeException(),
                        peerEofTrace()));
    }

    @ParameterizedTest
    @EnumSource(
            value = Status.Code.class,
            names = {"CANCELLED", "DEADLINE_EXCEEDED", "INTERNAL", "UNKNOWN"})
    void peerEofDoesNotOverrideOtherTerminalStatuses(Status.Code code) {
        assertFalse(
                EngineGrpcClient.shouldRetry(
                        AbstractGrpcClient.ServiceType.WORKER_STATUS,
                        Status.fromCode(code).asRuntimeException(),
                        peerEofTrace()));
    }

    @Test
    void peerEofDoesNotOverrideAMoreSpecificFailure() {
        assertFalse(
                EngineGrpcClient.shouldRetry(
                        AbstractGrpcClient.ServiceType.WORKER_STATUS,
                        Status.UNAVAILABLE
                                .withCause(new InvalidProtocolBufferException("invalid reply"))
                                .asRuntimeException(),
                        peerEofTrace()));
    }

    @ParameterizedTest
    @EnumSource(
            value = AbstractGrpcClient.ServiceType.class,
            names = {"BATCH_ENQUEUE", "ENGINE_CANCEL"})
    void peerEofDoesNotReplayNonReplayableOperations(AbstractGrpcClient.ServiceType service) {
        assertFalse(
                EngineGrpcClient.shouldRetry(
                        service, Status.UNAVAILABLE.asRuntimeException(), peerEofTrace()));
    }

    @Test
    void bareUnavailableWithoutSocketEvidenceDoesNotRetry() {
        GrpcCallTrace trace = mock(GrpcCallTrace.class);
        when(trace.snapshot()).thenReturn(GrpcCallTrace.Snapshot.EMPTY);
        assertFalse(
                EngineGrpcClient.shouldRetry(
                        AbstractGrpcClient.ServiceType.WORKER_STATUS,
                        Status.UNAVAILABLE.asRuntimeException(),
                        trace));
    }

    private static GrpcCallTrace peerEofTrace() {
        GrpcCallTrace trace = mock(GrpcCallTrace.class);
        when(trace.snapshot())
                .thenReturn(
                        new GrpcCallTrace.Snapshot(
                                GrpcCallTrace.Stage.WAITING_RESPONSE,
                                "PEER_EOF",
                                GrpcFailure.Kind.UNEXPECTED_EOF,
                                1,
                                -1,
                                0,
                                1,
                                -1,
                                -1,
                                10,
                                11,
                                9));
        return trace;
    }

    @Test
    void enqueueBatchDoesNotRetryAmbiguousBrokenConnection() {
        assertFalse(
                EngineGrpcClient.retriesBrokenConnections(
                        EngineGrpcClient.ServiceType.BATCH_ENQUEUE));
    }

    @Test
    void retryPolicyChangeDoesNotDisableReadOnlyRpcRetries() {
        assertTrue(
                EngineGrpcClient.retriesBrokenConnections(
                        EngineGrpcClient.ServiceType.WORKER_STATUS));
        assertTrue(
                EngineGrpcClient.retriesBrokenConnections(
                        EngineGrpcClient.ServiceType.CACHE_STATUS));
        assertTrue(
                EngineGrpcClient.retriesBrokenConnections(
                        EngineGrpcClient.ServiceType.MULTIMODAL_WORKER_STATUS));
        assertTrue(
                EngineGrpcClient.retriesBrokenConnections(
                        EngineGrpcClient.ServiceType.MULTIMODAL_CACHE_STATUS));
    }

    @Test
    void cancelKeepsItsExistingSingleShotPolicy() {
        assertFalse(
                EngineGrpcClient.retriesBrokenConnections(
                        EngineGrpcClient.ServiceType.ENGINE_CANCEL));
    }
}
