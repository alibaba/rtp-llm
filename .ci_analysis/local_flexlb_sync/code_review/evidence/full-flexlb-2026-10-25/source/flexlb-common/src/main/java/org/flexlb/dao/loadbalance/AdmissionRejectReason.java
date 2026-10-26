package org.flexlb.dao.loadbalance;

/**
 * Master-owned reason for rejecting an incoming admission request.
 *
 * <p>The value is deliberately independent of protobuf so the scheduling
 * domain does not depend on a transport adapter.  {@code UNSPECIFIED} means
 * that the Master cannot prove a more specific causal attribution from its
 * snapshot; it must never be reconstructed from an error string.</p>
 */
public enum AdmissionRejectReason {
    UNSPECIFIED,
    HIGHER_PRIORITY_AHEAD,
    SAME_PRIORITY_AHEAD,
    RESOURCE_EXHAUSTED
}
