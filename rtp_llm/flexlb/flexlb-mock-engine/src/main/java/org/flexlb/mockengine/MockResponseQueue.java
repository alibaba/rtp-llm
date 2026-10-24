package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;

import java.util.concurrent.LinkedBlockingQueue;

/** One client stream: after its terminal frame, late producers cannot write again. */
final class MockResponseQueue extends LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> {
    private boolean terminal;
    private long terminalNanos;

    synchronized boolean expiredTerminal(long now, long retentionMs) {
        return terminal && now - terminalNanos >= java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(retentionMs);
    }

    @Override
    public synchronized boolean offer(EngineRpcService.GenerateOutputsPB output) {
        if (terminal) {
            return false;
        }
        boolean finishes = output.hasErrorInfo()
                || output.getFlattenOutput().getFinishedList().contains(true);
        boolean accepted = super.offer(output);
        if (accepted && finishes) {
            terminal = true;
            terminalNanos = System.nanoTime();
        }
        return accepted;
    }
}
