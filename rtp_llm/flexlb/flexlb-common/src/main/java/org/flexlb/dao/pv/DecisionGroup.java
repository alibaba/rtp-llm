package org.flexlb.dao.pv;

/** One committed scheduling group, independent of engine batch IDs and later delivery outcomes. */
public record DecisionGroup(String id,
                            String policy,
                            String dispatcher,
                            String worker,
                            int committedSize,
                            String reason,
                            long committedAtMs,
                            long requestWaitMs) {
}
