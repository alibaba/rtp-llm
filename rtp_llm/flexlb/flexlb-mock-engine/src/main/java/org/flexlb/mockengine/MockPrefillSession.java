package org.flexlb.mockengine;

import java.util.concurrent.ScheduledFuture;

/**
 * The P-side RPC context, independent of the local prefill execution slot.
 * Batch Enqueue prepares Decode before ACK; Fetch attaches to this context.
 * Auto-fetch is the explicit benchmark shortcut through the same continuation.
 * Cross-engine callbacks run outside this monitor.
 */
final class MockPrefillSession {
    final MockPerformanceModel.RequestShape shape;
    final long batchId;
    final int dpRank;
    final boolean autoFetch;
    final JavaMockEngineCluster.FastRpcService decode;
    boolean remoteDecode;
    final long preparationStartedNanos;
    private boolean clientAttached;
    private boolean prefillDone;
    private boolean continued;
    private boolean closed;
    private org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB failure;
    private ScheduledFuture<?> expiry;

    MockPrefillSession(MockPerformanceModel.RequestShape shape, long batchId,
                       int dpRank, boolean autoFetch, JavaMockEngineCluster.FastRpcService decode) {
        this.shape = shape;
        this.batchId = batchId;
        this.dpRank = dpRank;
        this.autoFetch = autoFetch;
        this.decode = decode;
        this.preparationStartedNanos = System.nanoTime();
    }

    synchronized boolean attach() {
        if (closed || clientAttached) {
            return false;
        }
        clientAttached = true;
        cancelExpiry();
        return true;
    }

    synchronized void prefillDone() {
        prefillDone = true;
    }

    synchronized void fail(org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB output) {
        failure = output;
        prefillDone = true;
    }

    synchronized org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB failure() {
        return failure;
    }

    synchronized boolean claimContinuation() {
        if (closed || continued || !prefillDone || !(autoFetch || clientAttached)) {
            return false;
        }
        continued = true;
        cancelExpiry();
        return true;
    }

    synchronized boolean isClosed() {
        return closed;
    }

    synchronized boolean expireUnattached() {
        if (closed || autoFetch || clientAttached) {
            return false;
        }
        closed = true;
        return true;
    }

    synchronized void close() {
        closed = true;
        cancelExpiry();
    }

    synchronized void setExpiry(ScheduledFuture<?> future) {
        if (closed || autoFetch || clientAttached) {
            future.cancel(false);
        } else {
            expiry = future;
        }
    }

    private void cancelExpiry() {
        if (expiry != null) {
            expiry.cancel(false);
            expiry = null;
        }
    }
}
