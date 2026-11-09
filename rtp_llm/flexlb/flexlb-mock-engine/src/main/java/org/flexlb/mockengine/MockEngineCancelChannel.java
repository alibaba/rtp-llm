package org.flexlb.mockengine;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.engine.grpc.RequestId;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Resolves mock Prefill cancellation by gRPC port and returns accepted, seen-but-finished or
 * fenced-absent outcomes. Injected RPC faults leave engine state unchanged; successful cancellation
 * still requires worker terminal feedback before scheduler cleanup.
 */
public final class MockEngineCancelChannel implements EngineCancelChannel {

    private final Map<Integer, JavaMockEngineCluster.FastRpcService> services;

    public MockEngineCancelChannel(Map<Integer, JavaMockEngineCluster.FastRpcService> services) {
        this.services = services;
    }

    @Override
    public boolean isSupported(DecodeEndpoint endpoint) {
        return endpoint != null && services.containsKey(endpoint.getGrpcPort());
    }

    @Override
    public CompletableFuture<CancelAck> cancel(
            CancelTarget target, String requestId, long timeoutMs) {
        JavaMockEngineCluster.FastRpcService service =
                target == null ? null : services.get(target.prefillGrpcPort());
        if (service == null) {
            return CompletableFuture.completedFuture(CancelAck.UNSUPPORTED);
        }
        long engineRequestId;
        try {
            engineRequestId = RequestId.toEngineRequestId(requestId);
        } catch (IllegalArgumentException error) {
            return CompletableFuture.failedFuture(error);
        }
        // Cancel-RPC fault-injection gate — same arriveCancelRpc entry as the
        // gRPC Cancel handler and the HTTP /cancel_request surface: an armed
        // fault short-circuits BEFORE cancelRequest so the engine cancel
        // state machine is never touched (production semantics "RPC failed =
        // engine state unchanged": no fences, no terminal records, no census
        // branch) while the arrival stays counted.
        JavaMockEngineCluster.CancelFaultKind fault = service.arriveCancelRpc();
        if (fault == JavaMockEngineCluster.CancelFaultKind.NO_RESPOND) {
            // cancel_no_respond: never complete the future — the in-process
            // mirror of a hanging RPC; the caller's cancel-ack timeout fails it.
            return new CompletableFuture<>();
        }
        if (fault == JavaMockEngineCluster.CancelFaultKind.ERROR) {
            // cancel_error: transport-layer failure -> failed future, no
            // fence installed (mirrors the gRPC INTERNAL error path).
            return CompletableFuture.failedFuture(
                    new IllegalStateException("mock cancel error injection"));
        }
        if (fault == JavaMockEngineCluster.CancelFaultKind.UNEXPECTED_STATUS) {
            // cancel_unexpected_status: the RPC "succeeds" but the ack status
            // sits outside the contract — mirror the master-side response
            // mapping failing on the out-of-contract status instead of
            // accepting it (same terminal shape as HttpMockEngineCancel
            // Channel's unknown-status IllegalStateException).
            return CompletableFuture.failedFuture(
                    new IllegalStateException("unexpected cancel ack status"));
        }
        try {
            // Deliberately inspect only the addressed Prefill. Scanning other
            // workers would hide an incorrect Prefill route in tests.
            JavaMockEngineCluster.CancelResult result = service.cancelRequest(engineRequestId);
            if (result.found()) {
                return CompletableFuture.completedFuture(CancelAck.ACCEPTED);
            }
            // Production-faithful mapping (C++ Cancel handler): a request
            // this engine has seen but already finished answers NOT_FOUND
            // (the completion record stays deliverable from the retain
            // window); a never-seen rid answers REQUEST_FENCED — cancelRequest
            // installed the ABSENT_FENCE record that rejects any racing
            // later Enqueue of that rid with the typed 8429.
            return CompletableFuture.completedFuture(
                    result.alreadyFinished() ? CancelAck.NOT_FOUND : CancelAck.REQUEST_FENCED);
        } catch (UnsupportedOperationException e) {
            return CompletableFuture.completedFuture(CancelAck.FAILED);
        } catch (Exception e) {
            // Contract: never throw synchronously; surface as a failed future.
            return CompletableFuture.failedFuture(e);
        }
    }
}
