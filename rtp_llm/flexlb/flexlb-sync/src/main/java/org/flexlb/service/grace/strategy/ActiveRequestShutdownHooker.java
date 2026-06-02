package org.flexlb.service.grace.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.listener.AppShutDownHooker;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class ActiveRequestShutdownHooker implements AppShutDownHooker {

    public static volatile boolean shutdownCompletedSuccessfully;
    public static volatile long shutdownTimeoutMs = 300000;
    public static volatile long quietPeriodMs = 5000;

    private final ActiveRequestCounter activeRequestCounter;
    private final GracefulLifecycleReporter lifecycleReporter;

    public ActiveRequestShutdownHooker(ActiveRequestCounter activeRequestCounter,
                                       GracefulLifecycleReporter lifecycleReporter) {
        this.activeRequestCounter = activeRequestCounter;
        this.lifecycleReporter = lifecycleReporter;
    }

    @Override
    public void beforeShutdown() {
        log.info("ActiveRequestShutdownHooker: waiting for active requests to complete");
        long startTime = System.currentTimeMillis();
        long hardDeadline = startTime + shutdownTimeoutMs;

        try {
            boolean drained = awaitQuiet(hardDeadline);
            long duration = System.currentTimeMillis() - startTime;

            if (drained) {
                shutdownCompletedSuccessfully = true;
                lifecycleReporter.reportShutdownComplete(duration);
                log.info("ActiveRequestShutdownHooker: shutdown complete, total {}ms", duration);
            } else {
                shutdownCompletedSuccessfully = false;
                lifecycleReporter.reportShutdownTimeout(duration);
                log.error("ActiveRequestShutdownHooker: shutdown timeout after {}ms", duration);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            shutdownCompletedSuccessfully = false;
            log.error("ActiveRequestShutdownHooker: interrupted while waiting for requests to complete", e);
        }
    }

    /**
     * Wait until no active requests for a continuous quietPeriodMs.
     * Any active request resets the quiet deadline. Returns false if hardDeadline is exceeded.
     */
    private boolean awaitQuiet(long hardDeadline) throws InterruptedException {
        long quietDeadline = System.currentTimeMillis() + quietPeriodMs;

        while (System.currentTimeMillis() < quietDeadline) {
            if (System.currentTimeMillis() >= hardDeadline) {
                return false;
            }
            long count = activeRequestCounter.getCount();
            if (count > 0) {
                quietDeadline = System.currentTimeMillis() + quietPeriodMs;
                log.info("ActiveRequestShutdownHooker: activeCount={}, quiet deadline reset", count);
            }
            //noinspection BusyWait
            Thread.sleep(500);
        }
        return true;
    }
}
