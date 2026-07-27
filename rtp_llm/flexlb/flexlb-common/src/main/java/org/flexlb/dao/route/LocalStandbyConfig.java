package org.flexlb.dao.route;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

/**
 * Local cache metadata standby maintained from requests routed by FlexLB.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class LocalStandbyConfig {

    public static final long DEFAULT_ENTRY_TTL_MS = 300_000L;
    public static final long DEFAULT_MAXIMUM_ENTRIES = 2_000_000L;
    public static final double DEFAULT_CAPACITY_MULTIPLIER = 1.2;
    public static final int DEFAULT_ASYNC_QUEUE_CAPACITY = 100000;
    public static final int DEFAULT_HASH_THREAD_COUNT = 4;
    public static final int DEFAULT_HASH_QUEUE_CAPACITY = 100000;

    @JsonProperty("auto_switch")
    private boolean autoSwitch = true;

    /**
     * Local standby block size used only when hashes are calculated from input_ids. Zero reuses
     * the request's engine block size. Provided block_cache_keys always retain their request block
     * size.
     */
    @JsonProperty("block_size")
    private long blockSize;

    /**
     * Lifetime of one worker/block-hash association after its most recent write.
     */
    @JsonProperty("entry_ttl_ms")
    private long entryTtlMs = DEFAULT_ENTRY_TTL_MS;

    /**
     * Hard upper bound for worker/block-hash associations retained by this process.
     */
    @JsonProperty("maximum_entries")
    private long maximumEntries = DEFAULT_MAXIMUM_ENTRIES;

    /**
     * Headroom applied to the HBM block capacity reported by WorkerStatus.
     */
    @JsonProperty("capacity_multiplier")
    private double capacityMultiplier = DEFAULT_CAPACITY_MULTIPLIER;

    /**
     * Maximum number of pending request-recording tasks, not block hashes.
     */
    @JsonProperty("async_queue_capacity")
    private int asyncQueueCapacity = DEFAULT_ASYNC_QUEUE_CAPACITY;

    /**
     * Low-priority workers used only for Local Standby block hashes.
     */
    @JsonProperty("hash_thread_count")
    private int hashThreadCount = DEFAULT_HASH_THREAD_COUNT;

    /**
     * Maximum pending Local Standby hash tasks. The queue is deliberately smaller than the
     * metadata update queue because each task retains its request input_ids until execution.
     */
    @JsonProperty("hash_queue_capacity")
    private int hashQueueCapacity = DEFAULT_HASH_QUEUE_CAPACITY;

}
