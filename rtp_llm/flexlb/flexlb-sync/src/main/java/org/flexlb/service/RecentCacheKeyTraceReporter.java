package org.flexlb.service;

import com.google.common.math.LongMath;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRequirements;
import org.flexlb.cache.core.RecentCacheKeyWindow;
import org.flexlb.cache.monitor.CacheHitTheoryStats;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
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

    private static final String DEFAULT_MASTER_THEORY_LOG_PATH = "/home/admin/ai-whale/logs/master_theory_hit.log";
    private static final Object THEORY_LOG_LOCK = new Object();
    private static final DateTimeFormatter THEORY_LOG_TIME_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSXXX").withZone(ZoneId.systemDefault());

    @Autowired(required = false)
    private RecentCacheKeyWindow recentCacheKeyWindow;

    @Autowired(required = false)
    private CacheMetricsReporter cacheMetricsReporter;

    @Autowired(required = false)
    private ConfigService configService;

    private final CacheHitTheoryStats theoryStats = new CacheHitTheoryStats();

    private static volatile BufferedWriter theoryLogWriter;
    private static volatile boolean theoryLogOpenFailed;
    private static volatile boolean theoryLogFlushFailed;
    private static volatile String theoryLogPath = DEFAULT_MASTER_THEORY_LOG_PATH;

    private static final long FNV_OFFSET_BASIS = 0xcbf29ce484222325L;
    private static final long FNV_PRIME = 0x100000001b3L;

    public void report(RequestContext requestContext) {
        if (requestContext == null) {
            return;
        }
        FlexlbConfig config = requestContext.getConfig();
        if (config != null && !config.getObservability().getCacheHit().getRecentKeyWindow().isWriteEnabled()) {
            return;
        }

        Request request = requestContext.getRequest();
        RequestRequirements inputs = requestContext.getRequirements();
        if (inputs == null || recentCacheKeyWindow == null) {
            return;
        }

        List<Long> cacheKeys = inputs.blockCacheKeys();
        RecentCacheKeyWindow.Snapshot snapshot =
                recentCacheKeyWindow.record(cacheKeys);
        long inputTokens = Math.max(0L, inputs.seqLen());
        long hitTokens = theoryHitTokens(
                snapshot.getRequestHitOccurrences(),
                inputTokens,
                inputs.cacheKeyBlockSize());
        CacheHitTheoryStats.Snapshot theorySnapshot = theoryStats.record(
                hitTokens,
                inputTokens);
        logTraceIfEnabled(requestContext, request, inputs, snapshot, hitTokens, inputTokens, config);
        logTheoryIfEnabled(requestContext, inputs, theorySnapshot, config);

        if (cacheMetricsReporter == null || (config != null
                && !config.getObservability().getCacheHit().isMetricsEnabled())) {
            return;
        }

        cacheMetricsReporter.reportRecentCacheKeyHitMetrics(snapshot.getTimeWindowMs(),
                hitTokens,
                inputTokens);
        cacheMetricsReporter.reportTheoryCacheHitMetrics(theorySnapshot);
    }

    private static long theoryHitTokens(long hitKeyCount, long inputTokens, long cacheKeyBlockSize) {
        if (hitKeyCount <= 0L || inputTokens <= 0L || cacheKeyBlockSize <= 0L) {
            return 0L;
        }
        return Math.min(inputTokens, LongMath.saturatedMultiply(hitKeyCount, cacheKeyBlockSize));
    }

    @PostConstruct
    public void initializeTheoryLog() {
        FlexlbConfig config = configService == null ? null : configService.loadBalanceConfig();
        if (config == null || config.getObservability().getCacheHit().getTheoryLog() == null) {
            return;
        }
        theoryLogPath = config.getObservability().getCacheHit().getTheoryLog().getPath();
        synchronized (THEORY_LOG_LOCK) {
            getTheoryLogWriterLocked();
        }
    }

    private void logTraceIfEnabled(RequestContext requestContext,
                                   Request request,
                                   RequestRequirements inputs,
                                   RecentCacheKeyWindow.Snapshot snapshot,
                                   long hitTokens,
                                   long inputTokens,
                                   FlexlbConfig config) {
        if (config == null || !config.getObservability().getCacheHit().isRequestTraceLogEnabled()) {
            return;
        }
        List<Long> cacheKeys = inputs.blockCacheKeys();
        Logger.info("Master cache-key trace: masterRequestId={}, requestId={}, "
                        + "seqLen={}, requestTimeMs={}, requestCacheKeys={}, hitCacheKeys={}, hitRatio={}, "
                        + "hitTokens={}, inputTokens={}, tokenHitRatio={}, cacheKeyDigest={}, selectedServers={}, cacheKeys={}",
                requestContext.getRequestId(),
                inputs.requestId(),
                inputs.seqLen(),
                request == null ? 0L : request.getRequestTimeMs(),
                snapshot.getRequestOccurrences(),
                snapshot.getRequestHitOccurrences(),
                hitRatio(snapshot.getRequestHitOccurrences(), snapshot.getRequestOccurrences()),
                hitTokens,
                inputTokens,
                hitRatio(hitTokens, inputTokens),
                cacheKeyDigest(cacheKeys),
                formatServerStatusList(requestContext.getResponse()),
                cacheKeys == null ? "[]" : cacheKeys.toString());
    }

    private static double hitRatio(long hitCount, long totalCount) {
        if (totalCount <= 0L) {
            return 0.0D;
        }
        return (double) hitCount / totalCount;
    }

    private void logTheoryIfEnabled(RequestContext requestContext,
                                    RequestRequirements inputs,
                                    CacheHitTheoryStats.Snapshot snapshot,
                                    FlexlbConfig config) {
        if (config == null || config.getObservability().getCacheHit().getTheoryLog() == null) {
            return;
        }
        if (snapshot == null || snapshot.getRequestTotalCount() <= 0L) {
            return;
        }
        writeTheoryLogLine(formatTheoryLogLine(requestContext, inputs, snapshot));
    }

    private static String formatTheoryLogLine(RequestContext requestContext,
                                              RequestRequirements inputs,
                                              CacheHitTheoryStats.Snapshot snapshot) {
        return String.format(Locale.ROOT,
                "time=%s ts_ms=%d source=master master_request_id=%s request_id=%d seq_len=%d "
                        + "cache_key_block_size=%d request_hit_tokens=%d request_input_tokens=%d request_ratio=%.6f "
                        + "all_hit_tokens=%d all_input_tokens=%d all_ratio=%.6f",
                THEORY_LOG_TIME_FORMATTER.format(Instant.ofEpochMilli(snapshot.getNowMs())),
                snapshot.getNowMs(),
                requestContext == null ? "" : String.valueOf(requestContext.getRequestId()),
                inputs.requestId(),
                inputs.seqLen(),
                inputs.cacheKeyBlockSize(),
                snapshot.getRequestHitCount(),
                snapshot.getRequestTotalCount(),
                snapshot.getRequestHitRatio(),
                snapshot.getAllHitCount(),
                snapshot.getAllTotalCount(),
                snapshot.getAllHitRatio());
    }

    private static void writeTheoryLogLine(String line) {
        if (theoryLogOpenFailed) {
            return;
        }
        synchronized (THEORY_LOG_LOCK) {
            BufferedWriter writer = getTheoryLogWriterLocked();
            if (writer == null) {
                return;
            }
            try {
                writer.write(line);
                writer.newLine();
            } catch (IOException e) {
                Logger.warn("Failed to write master theory hit log: {}", e.getMessage());
            }
        }
    }

    private static BufferedWriter getTheoryLogWriterLocked() {
        if (theoryLogWriter != null || theoryLogOpenFailed) {
            return theoryLogWriter;
        }
        Path logPath = Path.of(theoryLogPath);
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

    @Scheduled(fixedDelay = 1000L)
    public void flushTheoryLog() {
        if (theoryLogWriter == null) {
            return;
        }
        synchronized (THEORY_LOG_LOCK) {
            BufferedWriter writer = theoryLogWriter;
            if (writer == null) {
                return;
            }
            try {
                writer.flush();
                if (theoryLogFlushFailed) {
                    theoryLogFlushFailed = false;
                    Logger.info("Master theory hit log flush recovered");
                }
            } catch (IOException e) {
                if (!theoryLogFlushFailed) {
                    theoryLogFlushFailed = true;
                    Logger.warn("Failed to flush master theory hit log: {}", e.getMessage());
                }
            }
        }
    }

    @PreDestroy
    public void closeTheoryLog() {
        synchronized (THEORY_LOG_LOCK) {
            if (theoryLogWriter == null) {
                return;
            }
            try {
                theoryLogWriter.close();
            } catch (IOException e) {
                Logger.warn("Failed to close master theory hit log: {}", e.getMessage());
            } finally {
                theoryLogWriter = null;
            }
        }
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
