package org.flexlb.dao;

import lombok.Data;
import lombok.ToString;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;

/**
 * Request lifecycle context for {@code /batch_schedule} endpoint.
 *
 * <p>Mirrors {@link BalanceContext} in role and shape, but models a different
 * operation: {@code /batch_schedule} is a single fire-and-forget decision that
 * fans out to {@code N} worker targets, with no queue, retry, or per-task
 * bookkeeping. The two operations only share incidental meter fields; they
 * deliberately do not share a base class until a third batch-style endpoint
 * makes the abstraction concrete.
 */
@Data
@ToString
public class BatchScheduleContext {

    //======================== Basic =======================//

    private FlexlbConfig config;

    private BatchScheduleRequest batchRequest;

    private BatchScheduleResponse batchResponse;

    //======================== Meters =======================//

    private long startTime = System.currentTimeMillis();

    private boolean success = true;

    private String errorMessage;
}
