package org.flexlb.cache.telemetry;

import org.springframework.stereotype.Component;

/**
 * Compatibility type for callers that still use the pre-refactor telemetry package.
 * The implementation and Spring bean now live in {@code cache.monitor}.
 */
@Deprecated
@Component
public class CacheMetricsReporter extends org.flexlb.cache.monitor.CacheMetricsReporter {}
