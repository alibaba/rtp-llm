package org.flexlb.balance.dp;

import java.util.List;

/**
 * Assigns each request in a batch to a dp_rank slot in [0, dpSize).
 *
 * Contract:
 *  - Returned RankAssignment list maps one-to-one with batch.requests() (any order)
 *  - Every dpRank MUST be in [0, batch.dpSize())
 *  - Implementations may be stateless or globally stateful (e.g. RR cursor) but
 *    MUST be thread-safe.
 */
public interface DpAssignStrategy {

    List<RankAssignment> assign(PrefillBatch batch);

    /** Strategy identifier, selected by FlexlbConfig.dpAssignStrategy. */
    String name();
}
