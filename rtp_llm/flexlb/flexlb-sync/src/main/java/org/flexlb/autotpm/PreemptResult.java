package org.flexlb.autotpm;

/**
 * Outcome of a confirmed preemption: the victim has verifiably released its
 * decode-side capacity, and the incoming request may be routed onto the freed
 * endpoint. Only produced by
 * {@link PriorityPressureController#tryPreempt} after the bounded release
 * wait succeeded — never optimistically.
 *
 * @param endpoint         decode endpoint key ({@code ip:httpPort}) whose
 *                         capacity was freed
 * @param victimRequestId  the preempted request
 * @param victimPriority   the preempted request's priority
 */
public record PreemptResult(String endpoint, long victimRequestId, int victimPriority) {
}
