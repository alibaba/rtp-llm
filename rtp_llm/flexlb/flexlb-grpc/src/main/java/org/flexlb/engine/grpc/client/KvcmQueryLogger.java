package org.flexlb.engine.grpc.client;

import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Records each returned host's local and global cache hit block counts.
 */
final class KvcmQueryLogger {

    private static final Logger QUERY_LOG = LoggerFactory.getLogger("kvcmLogger");
    private static final Logger LOG = LoggerFactory.getLogger(KvcmQueryLogger.class);

    private KvcmQueryLogger() {
    }

    static void logResponse(String requestId, GetHostCacheStateResponse response) {
        if (!QUERY_LOG.isInfoEnabled()) {
            return;
        }
        try {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("requestId", requestId);
            entry.put("hosts", response.getHostsList().stream()
                    .map(host -> Map.of("hostIpPort", host.getHostIpPort(),
                            "local", host.getLocal(), "global", host.getGlobal()))
                    .toList());
            QUERY_LOG.info("{}", JsonUtils.toString(entry));
        } catch (RuntimeException error) {
            LOG.warn("Failed to record KVCM query response, requestId={}", requestId, error);
        }
    }
}
