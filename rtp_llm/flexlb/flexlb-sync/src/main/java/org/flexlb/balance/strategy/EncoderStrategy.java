package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Prefers the least uncached Encoder work, breaking ties by concurrent requests.
 */
@Component
public final class EncoderStrategy implements EncoderSelectionStrategy {
    @Override
    public PlacementResult<Candidate, PlacementKey> select(
            List<? extends Candidate> candidates,
            String group,
            boolean weighted,
            int requestLimit) {
        Candidate winner = null;
        long leastTokens = Long.MAX_VALUE;
        long fewestRequests = Long.MAX_VALUE;
        for (Candidate candidate : candidates) {
            if (!candidate.available() || group != null && !group.equals(candidate.group()))
                continue;
            long requests = candidate.requestCount();
            if (requests >= requestLimit) continue;
            long tokens = weighted ? candidate.uncachedTokens() : 0L;
            if (tokens < leastTokens || tokens == leastTokens && requests < fewestRequests) {
                winner = candidate;
                leastTokens = tokens;
                fewestRequests = requests;
            }
        }
        return winner == null
                ? PlacementResult.blocked(
                        new PlacementKey(RoleType.ENCODER, group, null),
                        org.flexlb.dao.loadbalance.Response.error(
                                org.flexlb.dao.loadbalance.StrategyErrorType.NO_ENCODER_WORKER),
                        null)
                : PlacementResult.success(winner);
    }
}
