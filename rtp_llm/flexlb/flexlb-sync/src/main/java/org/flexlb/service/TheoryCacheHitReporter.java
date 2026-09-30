package org.flexlb.service;

import org.flexlb.cache.match.theory.RecentCacheKeyWindow;
import org.flexlb.cache.match.theory.TheoryCacheKeyHistory;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
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
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class TheoryCacheHitReporter {

    private static final Object THEORY_LOG_LOCK = new Object();
    private static final DateTimeFormatter THEORY_LOG_TIME_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSXXX").withZone(ZoneId.systemDefault());

    @Autowired(required = false)
    private TheoryCacheKeyHistory theoryCacheKeyHistory;

    @Autowired(required = false)
    private CacheMetricsReporter cacheMetricsReporter;

    @Autowired
    private ConfigService configService;

    private final AtomicLong droppedReportCount = new AtomicLong();
    private final ThreadPoolExecutor theoryHitReportExecutor = new ThreadPoolExecutor(
            1, 1, 0L, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(256), runnable -> {
                Thread thread = new Thread(runnable, "theory-cache-hit-reporter");
                thread.setDaemon(true);
                return thread;
            }, new ThreadPoolExecutor.AbortPolicy());

    private static volatile BufferedWriter theoryLogWriter;
    private static volatile boolean theoryLogOpenFailed;
    private static volatile boolean theoryLogFlushFailed;

    private static final long FNV_OFFSET_BASIS = 0xcbf29ce484222325L;
    private static final long FNV_PRIME = 0x100000001b3L;

    public void report(BalanceContext balanceContext) {
        if (balanceContext == null) {
            return;
        }
        try {
            theoryHitReportExecutor.execute(() -> {
                try {
                    reportRequest(balanceContext);
                } catch (RuntimeException failure) {
                    Logger.warn("Theory cache-hit report failed: request_id={}", balanceContext.getRequestId(), failure);
                }
            });
        } catch (RejectedExecutionException rejected) {
            droppedReportCount.incrementAndGet();
        }
    }

    private void reportRequest(BalanceContext balanceContext) {
        FlexlbConfig config = configService.loadBalanceConfig();
        if (!config.getObservability().getCacheHit().getRecentKeyWindow().isWriteEnabled()) {
            return;
        }
        Request request = balanceContext.getRequest();
        if (request == null || theoryCacheKeyHistory == null) {
            return;
        }
        RecentCacheKeyWindow.Snapshot snapshot = theoryCacheKeyHistory.record(request.getBlockCacheKeys());
        if (snapshot == null) {
            return;
        }
        long inputTokens = Math.max(0L, request.getSeqLen());
        long hitTokens = theoryHitTokens(snapshot.getRequestHitOccurrences(), inputTokens, request.getCacheKeyBlockSize());
        if (config.getObservability().getCacheHit().isRequestTraceLogEnabled()) {
            logTrace(balanceContext, snapshot, hitTokens, inputTokens);
        }
        if (theoryLogEnabled(config) && inputTokens > 0L) {
            writeTheoryLogLine(formatTheoryLogLine(balanceContext, hitTokens, inputTokens),
                    config.getObservability().getCacheHit().getTheoryLog().getPath());
        }
        if (cacheMetricsReporter != null && config.getObservability().getCacheHit().isMetricsEnabled()) {
            cacheMetricsReporter.reportTheoryCacheHitMetrics(hitTokens, inputTokens);
        }
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
        FlexlbConfig config = configService.loadBalanceConfig();
        if (!theoryLogEnabled(config)) {
            return;
        }
        synchronized (THEORY_LOG_LOCK) {
            getTheoryLogWriterLocked(config.getObservability().getCacheHit().getTheoryLog().getPath());
        }
    }

    private void logTrace(BalanceContext balanceContext, RecentCacheKeyWindow.Snapshot snapshot,
                          long hitTokens, long inputTokens) {
        Request request = balanceContext.getRequest();
        List<Long> cacheKeys = request.getBlockCacheKeys();
        Logger.info("Master cache-key trace: requestId={}, "
                        + "seqLen={}, requestCacheKeys={}, hitCacheKeys={}, hitRatio={}, "
                        + "hitTokens={}, inputTokens={}, tokenHitRatio={}, cacheKeyDigest={}, cacheKeys={}",
                balanceContext.getRequestId(), request.getSeqLen(),
                snapshot.getRequestOccurrences(), snapshot.getRequestHitOccurrences(),
                hitRatio(snapshot.getRequestHitOccurrences(), snapshot.getRequestOccurrences()),
                hitTokens, inputTokens, hitRatio(hitTokens, inputTokens), cacheKeyDigest(cacheKeys),
                formatCacheKeys(cacheKeys));
    }

    private static double hitRatio(long hitCount, long totalCount) {
        if (totalCount <= 0L) {
            return 0.0D;
        }
        return (double) hitCount / totalCount;
    }

    private static String formatTheoryLogLine(BalanceContext balanceContext, long hitTokens, long inputTokens) {
        Request request = balanceContext.getRequest();
        long nowMs = System.currentTimeMillis();
        return String.format(Locale.ROOT,
                "time=%s ts_ms=%d source=master request_id=%s seq_len=%d "
                        + "cache_key_block_size=%d request_hit_tokens=%d request_input_tokens=%d request_ratio=%.6f",
                formatTimestamp(nowMs),
                nowMs,
                balanceContext.getRequestId(),
                request.getSeqLen(),
                request.getCacheKeyBlockSize(),
                hitTokens,
                inputTokens,
                hitRatio(hitTokens, inputTokens));
    }

    private static String formatTimestamp(long timestampMs) {
        return THEORY_LOG_TIME_FORMATTER.format(Instant.ofEpochMilli(timestampMs));
    }

    private static void writeTheoryLogLine(String line, String logPath) {
        if (theoryLogOpenFailed) {
            return;
        }
        synchronized (THEORY_LOG_LOCK) {
            BufferedWriter writer = getTheoryLogWriterLocked(logPath);
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

    private static BufferedWriter getTheoryLogWriterLocked(String configuredPath) {
        if (theoryLogWriter != null || theoryLogOpenFailed) {
            return theoryLogWriter;
        }
        Path logPath = Path.of(configuredPath);
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

    private static boolean theoryLogEnabled(FlexlbConfig config) {
        return config.getObservability().getCacheHit().getTheoryLog() != null;
    }

    @Scheduled(fixedDelay = 1000L)
    public void flushTheoryLog() {
        long dropped = droppedReportCount.getAndSet(0L);
        if (dropped > 0L) {
            Logger.warn("Theory cache-hit reports dropped: count={}", dropped);
        }
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
        theoryHitReportExecutor.shutdown();
        try {
            if (!theoryHitReportExecutor.awaitTermination(5L, TimeUnit.SECONDS)) {
                theoryHitReportExecutor.shutdownNow();
            }
        } catch (InterruptedException interrupted) {
            theoryHitReportExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
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

    private static String formatCacheKeys(List<Long> cacheKeys) {
        return cacheKeys == null ? "[]" : cacheKeys.toString();
    }

}
