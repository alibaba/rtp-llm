package org.flexlb.dao.pv;

import lombok.Data;

/**
 * PV log entry for {@code /dispatcher/**} requests &mdash; both the batch fanout path
 * ({@code BatchHandler}) and the passthrough path ({@code PassthroughClient}) emit one record
 * per inbound request to {@code pvLogger}, sharing this schema so {@code pv.log} stays
 * single-format. (Those emitters live in {@code flexlb-api}; named here without {@code @link} so
 * this {@code flexlb-common} type carries no upward reference.)
 *
 * <p>Discriminated from {@link PvLogData} ({@code /schedule}) and {@link BatchPvLogData}
 * ({@code /batch_schedule}) by the {@code type} field. Batch-only counters are 0 for the
 * passthrough type; {@code feHost} is null for batch (which fans out to multiple FEs and
 * has no single "selected" host).
 *
 * <p>For passthrough, {@code costMs} captures the time from handler entry to FE response
 * headers, not body completion &mdash; streaming bodies are intentionally not waited on
 * (a 10-min SSE has no meaningful "request time"). For batch, {@code costMs} covers the
 * full fanout + merge.
 */
@Data
public class DispatchPvLogData implements PvRecord {

    public static final String TYPE_BATCH = "dispatch_batch";
    public static final String TYPE_PASSTHROUGH = "dispatch_passthrough";

    private String type;
    private String path;
    /** Caller-supplied model selector, bounded before assignment. Null when absent/non-scalar. */
    private String model;
    /** Caller correlation id ({@code __request_id__}, falling back to {@code request_id}). */
    private String callerRequestId;
    private int httpStatus;
    private boolean success;
    private long startTimeMs;
    private long costMs;

    /** Batch-only: canonical chunking configuration, e.g. {@code size:40} or {@code count:5}. */
    private String splitPolicy;
    /** Batch-only: total items in the inbound request array. 0 for passthrough. */
    private int totalItems;
    /** Batch-only: chunks fanned out (1 even for single-item batches). 0 for passthrough. */
    private int chunkCount;
    /** Batch-only: smallest emitted chunk. 0 before splitting or for an empty batch. */
    private int minChunkItems;
    /** Batch-only: largest emitted chunk. 0 before splitting or for an empty batch. */
    private int maxChunkItems;
    /** Batch-only: chunks that returned a non-2xx or threw. 0 for passthrough or all-OK batch. */
    private int failedChunks;

    /** Passthrough-only: the selected FE base URL. Null for batch. */
    private String feHost;

    /** Optional error message; null on success. */
    private String error;

    public static DispatchPvLogData batch(String path, long startTimeMs) {
        DispatchPvLogData d = new DispatchPvLogData();
        d.type = TYPE_BATCH;
        d.path = path;
        d.startTimeMs = startTimeMs;
        return d;
    }

    public static DispatchPvLogData passthrough(String path, long startTimeMs) {
        DispatchPvLogData d = new DispatchPvLogData();
        d.type = TYPE_PASSTHROUGH;
        d.path = path;
        d.startTimeMs = startTimeMs;
        return d;
    }

    public void finish(int httpStatus, String error) {
        this.httpStatus = httpStatus;
        this.success = httpStatus >= 200 && httpStatus < 300;
        this.error = error;
        this.costMs = System.currentTimeMillis() - startTimeMs;
    }
}
