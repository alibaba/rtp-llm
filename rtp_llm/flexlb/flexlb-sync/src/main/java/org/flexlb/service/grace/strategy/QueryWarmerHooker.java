package org.flexlb.service.grace.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.listener.AppOnlineHooker;
import org.flexlb.listener.ApplicationWarmupState;
import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.springframework.stereotype.Component;

import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
public class QueryWarmerHooker implements AppOnlineHooker, ApplicationWarmupState {

    private static final int WARMUP_WAIT_SECONDS = 10;

    private volatile boolean warmupFinished;
    private final GracefulLifecycleReporter lifecycleReporter;

    public QueryWarmerHooker(GracefulLifecycleReporter lifecycleReporter) {
        this.lifecycleReporter = lifecycleReporter;
    }

    @Override
    public void afterStartUp() {
        warmupFinished = false;
        Timer timer = new Timer("query-warmup-timeout", true);
        TimerTask task = new TimerTask() {
            @Override
            public void run() {
                warmupFinished = true;
                log.info("max wait time before health online finished");
            }
        };
        log.info("max wait time before health online: {} seconds", WARMUP_WAIT_SECONDS);
        timer.schedule(task, TimeUnit.SECONDS.toMillis(WARMUP_WAIT_SECONDS));
        try {
            doWarmUp();
        } finally {
            timer.cancel();
        }
    }

    @Override
    public int priority() {
        return 0;
    }

    /**
     * Warm up
     */
    private void doWarmUp() {
        log.info("do warm up: waiting for {} seconds for dependencies", WARMUP_WAIT_SECONDS);
        long startTime = System.currentTimeMillis();
        try {
            TimeUnit.SECONDS.sleep(WARMUP_WAIT_SECONDS);
            long duration = System.currentTimeMillis() - startTime;
            lifecycleReporter.reportWarmerComplete(duration);
            log.info("warm up success");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("warm up interrupted", e);
        } catch (Exception e) {
            log.error("warm up error", e);
        } finally {
            warmupFinished = true;
        }
    }

    @Override
    public boolean isWarmupFinished() {
        return warmupFinished;
    }
}
