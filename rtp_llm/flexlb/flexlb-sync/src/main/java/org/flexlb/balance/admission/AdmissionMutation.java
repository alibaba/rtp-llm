package org.flexlb.balance.admission;

import org.flexlb.dao.loadbalance.Response;

/** Exact one-shot ownership of one asynchronous admission mutation. */
public interface AdmissionMutation extends AutoCloseable {

    /**
     * Seal this exact mutation immediately before its first irreversible
     * admission side effect. A successful seal remains attached to this exact
     * mutation until it is resolved, so canonical publication may finish when
     * a later deadline, cancellation, or scheduler shutdown closes ordinary
     * admission.
     *
     * @return false when cancellation or deadline already closed admission;
     *         true when later cancellation must defer to this mutation
     */
    boolean seal();

    /** Transfer this exact mutation to canonical terminal ownership. */
    void terminate(Response failure);

    /** Complete a successful or side-effect-free mutation. */
    @Override
    void close();
}
