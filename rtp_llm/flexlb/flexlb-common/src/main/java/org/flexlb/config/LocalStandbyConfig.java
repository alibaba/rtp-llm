package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Local cache metadata standby maintained from requests routed by FlexLB. */
@Getter
@Setter
public final class LocalStandbyConfig {

    public static final long DEFAULT_TTL_MS = 300_000L;
    public static final long DEFAULT_MINIMUM_TTL_MS = 100_000L;
    public static final double DEFAULT_TTL_REDUCTION_START_RATIO = 0.8;
    public static final long DEFAULT_MAXIMUM_ENTRIES = 2_000_000L;
    public static final double DEFAULT_CAPACITY_MULTIPLIER = 10.0;
    public static final int DEFAULT_ASYNC_QUEUE_CAPACITY = 100_000;
    public static final int DEFAULT_HASH_THREAD_COUNT = 4;
    public static final int DEFAULT_HASH_QUEUE_CAPACITY = 100_000;

    private boolean autoSwitch = true;
    private long blockSize;
    private long ttlMs = DEFAULT_TTL_MS;
    private long minimumTtlMs = DEFAULT_MINIMUM_TTL_MS;
    private double ttlReductionStartRatio = DEFAULT_TTL_REDUCTION_START_RATIO;
    private long maximumEntries = DEFAULT_MAXIMUM_ENTRIES;
    private double capacityMultiplier = DEFAULT_CAPACITY_MULTIPLIER;
    private int asyncQueueCapacity = DEFAULT_ASYNC_QUEUE_CAPACITY;
    private int hashThreadCount = DEFAULT_HASH_THREAD_COUNT;
    private int hashQueueCapacity = DEFAULT_HASH_QUEUE_CAPACITY;
}
