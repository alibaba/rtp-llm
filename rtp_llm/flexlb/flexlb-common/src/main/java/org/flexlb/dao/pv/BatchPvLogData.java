package org.flexlb.dao.pv;

import lombok.Data;
import org.flexlb.dao.BatchScheduleContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;

import java.util.List;

/**
 * PV log entry for {@code /batch_schedule} requests.
 *
 * <p>Separate from {@link PvLogData} (used by {@code /schedule}) because the
 * two endpoints have different cardinality and field semantics. The {@code type}
 * field allows downstream log consumers to discriminate between the two schemas
 * sharing the same {@code pv.log} file.
 */
@Data
public class BatchPvLogData {

    private String type = "batch_schedule";
    private int batchCount;
    private int targetCount;
    private boolean success;
    private int code;
    private String error;
    private long startTimeMs;
    private long costMs;

    public BatchPvLogData(BatchScheduleContext bctx) {
        BatchScheduleRequest request = bctx.getBatchRequest();
        BatchScheduleResponse response = bctx.getBatchResponse();

        this.batchCount = request != null ? request.getBatchCount() : 0;
        this.startTimeMs = bctx.getStartTime();
        this.costMs = System.currentTimeMillis() - bctx.getStartTime();
        this.success = bctx.isSuccess();
        this.error = bctx.getErrorMessage();

        if (response != null) {
            this.code = response.getCode();
            List<BatchScheduleTarget> targets = response.getServerStatus();
            this.targetCount = targets != null ? targets.size() : 0;
        }
    }
}
