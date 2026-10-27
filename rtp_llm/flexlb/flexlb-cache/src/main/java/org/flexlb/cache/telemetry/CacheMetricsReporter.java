package org.flexlb.cache.telemetry;

/**
 * Compatibility type for callers that still use the pre-refactor telemetry package.
 * The implementation and Spring bean now live in {@code cache.monitor}.
 */
@Deprecated
public class CacheMetricsReporter extends org.flexlb.cache.monitor.CacheMetricsReporter {}
