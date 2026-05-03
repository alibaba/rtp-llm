package org.flexlb.dao.master;

/**
 * Per-DP-rank slice of {@link WorkerStatus}, carried in the {@code dp_status[]}
 * field of {@code WorkerStatusPB} and populated by DP0 when {@code dp_size > 1}.
 *
 * <p>Master treats the outer {@link WorkerStatus} as the aggregate-of-pod view
 * (sum / max / min / AND / union); per-rank breakdown lives here for V2
 * rank-aware routing. V1 routing does NOT consume these fields — they are
 * preserved purely as a data foundation.
 */
public record DpRankStatus(int dpRank,
                           String ip,
                           int grpcPort,
                           int availableConcurrency,
                           int runningQueryLen,
                           int waitingQueryLen,
                           int iterateCount,
                           boolean alive) {
}
