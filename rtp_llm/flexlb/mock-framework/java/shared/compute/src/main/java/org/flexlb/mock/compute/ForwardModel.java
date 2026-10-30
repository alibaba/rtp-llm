package org.flexlb.mock.compute;

import java.time.Duration;

/** Implementations see the full batch and return base computation time only. */
@FunctionalInterface
public interface ForwardModel {
    Duration estimate(ForwardBatch batch);
}
