package org.flexlb.balance.dp;

/** dp_rank assigned by Master to a single request. */
public record RankAssignment(PendingRequest request, int dpRank) {
}
