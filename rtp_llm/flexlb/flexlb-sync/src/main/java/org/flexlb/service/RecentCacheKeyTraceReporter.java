package org.flexlb.service;

import org.flexlb.cache.core.RecentCacheKeyWindow;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class RecentCacheKeyTraceReporter {

    @Autowired(required = false)
    private RecentCacheKeyWindow recentCacheKeyWindow;

    @Autowired(required = false)
    private CacheMetricsReporter cacheMetricsReporter;

    private static final long FNV_OFFSET_BASIS = 0xcbf29ce484222325L;
    private static final long FNV_PRIME = 0x100000001b3L;

    public void report(BalanceContext balanceContext) {
        if (balanceContext == null) {
            return;
        }
        FlexlbConfig config = balanceContext.getConfig();
        if (config != null && !config.isCacheHitWindowWriteEnabled()) {
            return;
        }

        Request request = balanceContext.getRequest();
        if (request == null || recentCacheKeyWindow == null) {
            return;
        }

        RecentCacheKeyWindow.Snapshot snapshot = recentCacheKeyWindow.record(request.getBlockCacheKeys());
        logTraceIfEnabled(balanceContext, request, snapshot, config);

        if (cacheMetricsReporter == null || (config != null && !config.isCacheHitMetricReportEnabled())) {
            return;
        }

        cacheMetricsReporter.reportRecentCacheKeyHitMetrics(snapshot.getTimeWindowMs(),
                snapshot.getRequestHitOccurrences(),
                snapshot.getRequestOccurrences());
    }

    private void logTraceIfEnabled(BalanceContext balanceContext,
                                   Request request,
                                   RecentCacheKeyWindow.Snapshot snapshot,
                                   FlexlbConfig config) {
        if (config == null || !config.isCacheHitTraceLogEnabled()) {
            return;
        }
        List<Long> cacheKeys = request.getBlockCacheKeys();
        Logger.info("Master cache-key trace: masterRequestId={}, requestId={}, retryCount={}, "
                        + "seqLen={}, requestTimeMs={}, requestCacheKeys={}, hitCacheKeys={}, hitRatio={}, "
                        + "cacheKeyDigest={}, selectedServers={}, cacheKeys={}",
                balanceContext.getRequestId(),
                request.getRequestId(),
                balanceContext.getRetryCount(),
                request.getSeqLen(),
                request.getRequestTimeMs(),
                snapshot.getRequestOccurrences(),
                snapshot.getRequestHitOccurrences(),
                hitRatio(snapshot.getRequestHitOccurrences(), snapshot.getRequestOccurrences()),
                cacheKeyDigest(cacheKeys),
                formatServerStatusList(balanceContext.getResponse()),
                formatCacheKeys(cacheKeys));
    }

    private static double hitRatio(long hitCount, long totalCount) {
        if (totalCount <= 0L) {
            return 0.0D;
        }
        return (double) hitCount / totalCount;
    }

    private static String cacheKeyDigest(List<Long> cacheKeys) {
        long digest = FNV_OFFSET_BASIS;
        if (cacheKeys == null || cacheKeys.isEmpty()) {
            return Long.toUnsignedString(digest);
        }
        for (Long cacheKey : cacheKeys) {
            if (cacheKey == null) {
                continue;
            }
            long value = cacheKey;
            digest ^= value;
            digest *= FNV_PRIME;
            digest ^= value >>> 32;
            digest *= FNV_PRIME;
        }
        return Long.toUnsignedString(digest);
    }

    private static String formatCacheKeys(List<Long> cacheKeys) {
        return cacheKeys == null ? "[]" : cacheKeys.toString();
    }

    private static String formatServerStatusList(Response response) {
        if (response == null || response.getServerStatus() == null || response.getServerStatus().isEmpty()) {
            return "[]";
        }
        StringBuilder builder = new StringBuilder("[");
        List<ServerStatus> serverStatusList = response.getServerStatus();
        for (int i = 0; i < serverStatusList.size(); i++) {
            if (i > 0) {
                builder.append(", ");
            }
            ServerStatus status = serverStatusList.get(i);
            if (status == null) {
                builder.append("null");
                continue;
            }
            builder.append(status.getRole())
                    .append("@")
                    .append(status.getServerIp())
                    .append(":")
                    .append(status.getGrpcPort())
                    .append("/http:")
                    .append(status.getHttpPort())
                    .append(",group=")
                    .append(status.getGroup())
                    .append(",success=")
                    .append(status.isSuccess())
                    .append(",code=")
                    .append(status.getCode())
                    .append(",message=")
                    .append(status.getMessage());
        }
        return builder.append("]").toString();
    }
}
