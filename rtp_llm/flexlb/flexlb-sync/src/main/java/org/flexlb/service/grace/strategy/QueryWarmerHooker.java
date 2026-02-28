package org.flexlb.service.grace.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.listener.AppOnlineHooker;
import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.flexlb.service.grace.GracefulOnlineService;
import org.springframework.stereotype.Component;

import java.util.Timer;
import java.util.TimerTask;

@Slf4j
@Component
public class QueryWarmerHooker implements AppOnlineHooker {

    public static boolean warmUpFinished;
    private static final int maxWaitTimeSeconds = 3;
    private final GracefulLifecycleReporter lifecycleReporter;

    public QueryWarmerHooker(GracefulLifecycleReporter lifecycleReporter) {
        this.lifecycleReporter = lifecycleReporter;
        GracefulOnlineService.addOnlineListener(this);
    }

    @Override
    public void afterStartUp() {

        // Set maximum warm-up wait time
        Timer timer = new Timer();
        TimerTask task = new TimerTask() {
            @Override
            public void run() {
                QueryWarmerHooker.warmUpFinished = true;
                log.info("max wait time before health online finished");
            }
        };
        log.info("max wait time before health online: {}", maxWaitTimeSeconds);
        timer.schedule(task, maxWaitTimeSeconds * 1000); // Execute after delayTime seconds

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
        log.info("do warm up: waiting for 3 seconds for sync engine");
        long startTime = System.currentTimeMillis();
        try {
            Thread.sleep(3000);
            long duration = System.currentTimeMillis() - startTime;
            lifecycleReporter.reportWarmerComplete(duration);
            log.info("warm up success");
        } catch (Exception e) {
            log.error("warm up error", e);
        } finally {
            warmUpFinished = true;
        }
    }

}
