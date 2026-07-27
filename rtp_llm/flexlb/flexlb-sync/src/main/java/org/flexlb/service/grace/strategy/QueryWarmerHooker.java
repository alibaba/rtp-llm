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
        long startTime = System.currentTimeMillis();
        Timer timer = new Timer("query-warmup-timeout", true);
        TimerTask task = new TimerTask() {
            @Override
            public void run() {
                try {
                    long duration = System.currentTimeMillis() - startTime;
                    lifecycleReporter.reportWarmerComplete(duration);
                    log.info("warm up success");
                } catch (Exception e) {
                    log.error("warm up error", e);
                } finally {
                    warmupFinished = true;
                    timer.cancel();
                }
            }
        };
        log.info("max wait time before health online: {} seconds", WARMUP_WAIT_SECONDS);
        timer.schedule(task, TimeUnit.SECONDS.toMillis(WARMUP_WAIT_SECONDS));
    }

    @Override
    public int priority() {
        return 0;
    }

    @Override
    public boolean isWarmupFinished() {
        return warmupFinished;
    }
}
