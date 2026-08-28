package org.flexlb.balance.admission;

import org.flexlb.dao.loadbalance.Response;

/** Exact one-shot ownership of one asynchronous admission mutation. */
public interface AdmissionMutation extends AutoCloseable {

    /** Transfer this exact mutation to canonical terminal ownership. */
    void terminate(Response failure);

    /** Complete a successful or side-effect-free mutation. */
    @Override
    void close();
}
