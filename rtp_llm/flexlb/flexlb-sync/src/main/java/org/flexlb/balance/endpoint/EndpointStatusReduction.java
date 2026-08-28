package org.flexlb.balance.endpoint;

/**
 * Immutable, response-local facts emitted by one canonical endpoint status
 * reduction. Implementations carry exact endpoint-owned identities rather than
 * exposing the raw worker observation to downstream consumers.
 */
public sealed interface EndpointStatusReduction
        permits EndpointStatusReduction.None,
                PrefillEndpoint.StatusReduction,
                DecodeEndpoint.StatusReduction {

    static EndpointStatusReduction none() {
        return None.INSTANCE;
    }

    /** Roles without scheduler-visible request ownership emit no facts. */
    enum None implements EndpointStatusReduction {
        INSTANCE
    }
}
