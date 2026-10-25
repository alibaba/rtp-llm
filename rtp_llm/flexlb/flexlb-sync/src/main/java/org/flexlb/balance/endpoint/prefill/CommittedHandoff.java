package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.endpoint.EndpointGenerationLifecycle;
import org.flexlb.balance.estimate.WorkView;

/**
 * One-shot capability returned when an OPEN admission commits.
 */
public final class CommittedHandoff implements AutoCloseable {
    public EndpointGenerationLifecycle.HandoffPermit generationHandoff;
    private final WorkView precedingWork;

    public CommittedHandoff(EndpointGenerationLifecycle.HandoffPermit generationHandoff, WorkView precedingWork) {
        this.generationHandoff = generationHandoff;
        this.precedingWork = precedingWork;
    }

    /**
     * Other reserved work at the same ownership boundary that committed this admission.
     */
    public WorkView precedingWork() {
        return precedingWork;
    }

    @Override
    public synchronized void close() {
        EndpointGenerationLifecycle.HandoffPermit exact = generationHandoff;
        if (exact == null) {
            return;
        }
        generationHandoff = null;
        exact.close();
    }
}
