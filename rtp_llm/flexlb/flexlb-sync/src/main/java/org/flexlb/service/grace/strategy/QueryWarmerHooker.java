package org.flexlb.service.grace.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.listener.AppOnlineHooker;
import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class QueryWarmerHooker implements AppOnlineHooker {

    public static volatile boolean warmUpFinished;
    private static final long DEFAULT_WARM_UP_WAIT_MS = 3_000L;
    private final GracefulLifecycleReporter lifecycleReporter;
    private final long warmUpWaitMs;

    public QueryWarmerHooker(GracefulLifecycleReporter lifecycleReporter) {
        this(lifecycleReporter, DEFAULT_WARM_UP_WAIT_MS);
    }

    QueryWarmerHooker(
            GracefulLifecycleReporter lifecycleReporter,
            long warmUpWaitMs) {
        this.lifecycleReporter = lifecycleReporter;
        if (warmUpWaitMs < 0L) {
            throw new IllegalArgumentException("warmUpWaitMs must not be negative");
        }
        this.warmUpWaitMs = warmUpWaitMs;
    }

    @Override
    public void afterStartUp() {
        warmUpFinished = false;
        doWarmUp();
    }

    @Override
    public int priority() {
        return 0;
    }

    /**
     * Warm up
     */
    private void doWarmUp() {
        log.info("do warm up: waiting for {} ms for sync engine", warmUpWaitMs);
        long startTime = System.currentTimeMillis();
        try {
            Thread.sleep(warmUpWaitMs);
            long duration = System.currentTimeMillis() - startTime;
            lifecycleReporter.reportWarmerComplete(duration);
            log.info("warm up success");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("warm up interrupted", e);
        } catch (Exception e) {
            log.error("warm up error", e);
        } finally {
            warmUpFinished = true;
        }
    }

}
