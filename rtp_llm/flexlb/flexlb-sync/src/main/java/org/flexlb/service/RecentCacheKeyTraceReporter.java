package org.flexlb.service;

import org.flexlb.cache.core.RecentCacheKeyWindow;
import org.flexlb.cache.core.ShardedRecentCacheKeyWindow;
import org.flexlb.cache.monitor.CacheHitTheoryStats;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
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

    private static final String CACHE_HIT_THEORY_LOG_PATH_ENV = "CACHE_HIT_THEORY_LOG_PATH";
    private static final String DEFAULT_MASTER_THEORY_LOG_PATH = "/home/admin/ai-whale/logs/master_theory_hit.log";
    private static final Object THEORY_LOG_LOCK = new Object();
    private static final DateTimeFormatter THEORY_LOG_TIME_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSXXX").withZone(ZoneId.systemDefault());

    @Autowired(required = false)
    private ShardedRecentCacheKeyWindow shardedRecentCacheKeyWindow;

    @Autowired(required = false)
    private CacheMetricsReporter cacheMetricsReporter;

    @Autowired(required = false)
    private ConfigService configService;

    private final CacheHitTheoryStats theoryStats = new CacheHitTheoryStats();

    private static volatile BufferedWriter theoryLogWriter;
    private static volatile boolean theoryLogOpenFailed;

    public void report(BalanceContext balanceContext) {
        if (balanceContext == null) {
            return;
        }

        Request request = balanceContext.getRequest();
        if (request == null || shardedRecentCacheKeyWindow == null) {
            return;
        }

        List<Long> cacheKeys = request.getBlockCacheKeys();
        RecentCacheKeyWindow.Snapshot snapshot =
                shardedRecentCacheKeyWindow.record(balanceContext.getRequestId(), cacheKeys);
        long inputTokens = Math.max(0L, request.getSeqLen());
        long hitTokens = theoryHitTokens(
                snapshot.getRequestHitOccurrences(),
                inputTokens,
                request.getCacheKeyBlockSize());
        CacheHitTheoryStats.Snapshot theorySnapshot = theoryStats.record(
                hitTokens,
                inputTokens);
        logTheoryIfEnabled(balanceContext, request, theorySnapshot);

        if (cacheMetricsReporter == null) {
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
        long hitTokens = hitKeyCount * cacheKeyBlockSize;
        if (hitTokens < 0L) {
            return inputTokens;
        }
        return Math.min(inputTokens, hitTokens);
    }

    @PostConstruct
    public void initializeTheoryLog() {
        if (configService == null) {
            return;
        }
        synchronized (THEORY_LOG_LOCK) {
            getTheoryLogWriterLocked();
        }
    }

    private void logTheoryIfEnabled(BalanceContext balanceContext,
                                    Request request,
                                    CacheHitTheoryStats.Snapshot snapshot) {
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
            } catch (IOException e) {
                Logger.warn("Failed to flush master theory hit log: {}", e.getMessage());
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
}
