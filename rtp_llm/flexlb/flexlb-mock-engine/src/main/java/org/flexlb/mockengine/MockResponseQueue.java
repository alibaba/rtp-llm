package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;

import java.util.concurrent.LinkedBlockingQueue;

/** One client stream: after its terminal frame, late producers cannot write again. */
final class MockResponseQueue extends LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> {
    private final Runnable onTerminal;
    MockResponseQueue() { this(() -> {}); }
    MockResponseQueue(Runnable onTerminal) { this.onTerminal = onTerminal; }
    private boolean terminal;
    private long terminalNanos;

    synchronized boolean expiredTerminal(long now, long retentionMs) {
        return terminal && now - terminalNanos >= java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(retentionMs);
    }

    @Override
    public boolean offer(EngineRpcService.GenerateOutputsPB output) {
        boolean finishes;
        synchronized (this) {
            if (terminal) return false;
            finishes = output.hasErrorInfo() || output.getFlattenOutput().getFinishedList().contains(true);
            if (!super.offer(output)) return false;
            if (finishes) {
                terminal = true;
                terminalNanos = System.nanoTime();
            }
        }
        if (finishes) onTerminal.run();
        return true;
    }
}
