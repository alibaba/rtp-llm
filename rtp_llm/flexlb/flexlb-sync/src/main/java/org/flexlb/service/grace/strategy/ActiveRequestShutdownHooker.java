package org.flexlb.service.grace.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.listener.AppShutDownHooker;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.flexlb.service.grace.GracefulShutdownService;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class ActiveRequestShutdownHooker implements AppShutDownHooker {

    public static boolean shutdownCompletedSuccessfully;
    public static long shutdownTimeoutMs = 300000;
    private final ActiveRequestCounter activeRequestCounter;
    private final GracefulLifecycleReporter lifecycleReporter;

    public ActiveRequestShutdownHooker(ActiveRequestCounter activeRequestCounter,
                                       GracefulLifecycleReporter lifecycleReporter) {
        this.activeRequestCounter = activeRequestCounter;
        this.lifecycleReporter = lifecycleReporter;
        GracefulShutdownService.addShutdownListener(this);
    }

    @Override
    public void beforeShutdown() {
        log.info("ActiveRequestShutdownHooker: waiting for active requests to complete");
        long startTime = System.currentTimeMillis();
        long pollIntervalMs = 1000;
        int repeatCount = 0;

        try {
            while (System.currentTimeMillis() - startTime < shutdownTimeoutMs) {
                if (activeRequestCounter.getCount() <= 0) {
                    long duration = System.currentTimeMillis() - startTime;
                    lifecycleReporter.reportShutdownComplete(duration);
                    shutdownCompletedSuccessfully = true;
                    log.info("ActiveRequestShutdownHooker: shutdown complete, all requests finished");
                    return;
                }
                log.info("waiting for activeRequestCounter to zero: {}, repeatCount: {}",
                        activeRequestCounter.getCount(), repeatCount);
                repeatCount++;
                //noinspection BusyWait
                Thread.sleep(pollIntervalMs);
            }

            long duration = System.currentTimeMillis() - startTime;
            lifecycleReporter.reportShutdownTimeout(duration);
            shutdownCompletedSuccessfully = false;
            log.error("ActiveRequestShutdownHooker: shutdown timeout, active requests still pending");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            shutdownCompletedSuccessfully = false;
            log.error("ActiveRequestShutdownHooker: interrupted while waiting for requests to complete", e);
        }
    }
}