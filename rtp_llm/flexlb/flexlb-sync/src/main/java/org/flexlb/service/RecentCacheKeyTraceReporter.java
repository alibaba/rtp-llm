package org.flexlb.service;

import org.flexlb.cache.core.RecentCacheKeyWindow;
import org.flexlb.cache.monitor.CacheHitTheoryStats;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;

@Component
public class RecentCacheKeyTraceReporter {

    private static final String CACHE_HIT_THEORY_LOG_PATH_ENV = "CACHE_HIT_THEORY_LOG_PATH";
    private static final String DEFAULT_MASTER_THEORY_LOG_PATH = "/home/admin/ai-whale/logs/master_theory_hit.log";
    private static final Object THEORY_LOG_LOCK = new Object();
    private static final DateTimeFormatter THEORY_LOG_TIME_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSXXX").withZone(ZoneId.systemDefault());

    @Autowired(required = false)
    private RecentCacheKeyWindow recentCacheKeyWindow;

    @Autowired(required = false)
    private CacheMetricsReporter cacheMetricsReporter;

    private final CacheHitTheoryStats theoryStats = new CacheHitTheoryStats();

    private static BufferedWriter theoryLogWriter;
    private static boolean theoryLogOpenFailed;

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

        List<Long> cacheKeys = request.getBlockCacheKeys();
        RecentCacheKeyWindow.Snapshot snapshot = recentCacheKeyWindow.record(cacheKeys);
        long inputTokens = Math.max(0L, request.getSeqLen());
        long hitTokens = theoryHitTokens(
                snapshot.getRequestHitOccurrences(),
                inputTokens,
                request.getCacheKeyBlockSize());
        CacheHitTheoryStats.Snapshot theorySnapshot = theoryStats.record(
                hitTokens,
                inputTokens);
        logTraceIfEnabled(balanceContext, request, snapshot, config);
        logTheoryIfEnabled(balanceContext, request, theorySnapshot, config);

        if (cacheMetricsReporter == null || (config != null && !config.isCacheHitMetricReportEnabled())) {
            return;
        }

        cacheMetricsReporter.reportRecentCacheKeyHitMetrics(snapshot.getTimeWindowMs(),
                snapshot.getRequestHitOccurrences(),
                snapshot.getRequestOccurrences());
        cacheMetricsReporter.reportTheoryCacheHitMetrics(theorySnapshot);
    }

    private static long theoryHitTokens(long hitKeyCount, long inputTokens, long cacheKeyBlockSize) {
        if (hitKeyCount <= 0L || inputTokens <= 0L || cacheKeyBlockSize <= 0L) {
            return 0L;
        }
        long hitTokens = hitKeyCount * cacheKeyBlockSize;
        if (hitTokens < 0L) {
            return inputTokens;
        }
        return Math.min(inputTokens, hitTokens);
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

    private void logTheoryIfEnabled(BalanceContext balanceContext,
                                    Request request,
                                    CacheHitTheoryStats.Snapshot snapshot,
                                    FlexlbConfig config) {
        if (config == null || !config.isCacheHitTheoryLogEnabled()) {
            return;
        }
        if (snapshot == null || snapshot.getRequestTotalCount() <= 0L) {
            return;
        }
        writeTheoryLogLine(formatTheoryLogLine(balanceContext, request, snapshot));
    }

    private static String formatTheoryLogLine(BalanceContext balanceContext,
                                              Request request,
                                              CacheHitTheoryStats.Snapshot snapshot) {
        return String.format(Locale.ROOT,
                "time=%s ts_ms=%d source=master master_request_id=%s request_id=%d seq_len=%d "
                        + "cache_key_block_size=%d request_hit_tokens=%d request_input_tokens=%d request_ratio=%.6f "
                        + "all_hit_tokens=%d all_input_tokens=%d all_ratio=%.6f",
                formatTimestamp(snapshot.getNowMs()),
                snapshot.getNowMs(),
                balanceContext == null ? "" : String.valueOf(balanceContext.getRequestId()),
                request == null ? 0L : request.getRequestId(),
                request == null ? 0L : request.getSeqLen(),
                request == null ? 0L : request.getCacheKeyBlockSize(),
                snapshot.getRequestHitCount(),
                snapshot.getRequestTotalCount(),
                snapshot.getRequestHitRatio(),
                snapshot.getAllHitCount(),
                snapshot.getAllTotalCount(),
                snapshot.getAllHitRatio());
    }

    private static String formatTimestamp(long timestampMs) {
        return THEORY_LOG_TIME_FORMATTER.format(Instant.ofEpochMilli(timestampMs));
    }

    private static void writeTheoryLogLine(String line) {
        synchronized (THEORY_LOG_LOCK) {
            BufferedWriter writer = getTheoryLogWriterLocked();
            if (writer == null) {
                return;
            }
            try {
                writer.write(line);
                writer.newLine();
                writer.flush();
            } catch (IOException e) {
                Logger.warn("Failed to write master theory hit log: {}", e.getMessage());
            }
        }
    }

    private static BufferedWriter getTheoryLogWriterLocked() {
        if (theoryLogWriter != null || theoryLogOpenFailed) {
            return theoryLogWriter;
        }
        String configuredPath = System.getenv(CACHE_HIT_THEORY_LOG_PATH_ENV);
        Path logPath = Path.of(configuredPath == null || configuredPath.isBlank() ?
                DEFAULT_MASTER_THEORY_LOG_PATH : configuredPath);
        try {
            Path parent = logPath.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            theoryLogWriter = Files.newBufferedWriter(logPath,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
            Logger.info("Master theory hit log path: {}", logPath);
        } catch (IOException e) {
            theoryLogOpenFailed = true;
            Logger.warn("Failed to open master theory hit log path {}: {}", logPath, e.getMessage());
        }
        return theoryLogWriter;
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
