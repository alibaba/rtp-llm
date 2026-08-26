package org.flexlb.balance.scheduler;

import java.util.concurrent.CompletableFuture;
import java.util.function.BooleanSupplier;

/** Narrow exact-slot commit capability used by route placement. */
interface InflightCommitPort {
    boolean commitInflight(
            BatchItem item,
            boolean priorityAdmission,
            BooleanSupplier commitAction);

    boolean isInflightGeneration(
            long requestId,
            CompletableFuture<?> future);
}
