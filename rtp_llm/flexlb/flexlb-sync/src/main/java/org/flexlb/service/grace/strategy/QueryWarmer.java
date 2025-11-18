package org.flexlb.service.grace.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.listener.OnlineListener;
import org.flexlb.service.grace.GracefulOnlineService;
import org.springframework.stereotype.Component;

import java.util.Timer;
import java.util.TimerTask;

@Slf4j
@Component
public class QueryWarmer implements OnlineListener {

    public static boolean warmUpFinished;
    private static final int maxWaitTimeSeconds = 120;

    public QueryWarmer() {
        GracefulOnlineService.addOnlineListener(this);
    }

    @Override
    public void afterStartUp() {

        // 设置最大预热等待时间
        Timer timer = new Timer();
        TimerTask task = new TimerTask() {
            @Override
            public void run() {
                QueryWarmer.warmUpFinished = true;
                log.info("max wait time before health online finished");
            }
        };
        log.info("max wait time before health online: {}", maxWaitTimeSeconds);
        timer.schedule(task, maxWaitTimeSeconds * 1000); // 延迟delayTime秒后执行

        doWarmUp();
    }

    @Override
    public int priority() {
        return 0;
    }

    /**
     * 预热
     */
    private void doWarmUp() {
        log.info("do warm up: waiting for 120 seconds for sync engine");
        try {
            Thread.sleep(120000); // 等待120秒
            log.info("warm up success");
        } catch (InterruptedException e) {
            log.warn("warm up interrupted", e);
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            log.error("warm up error", e);
        } finally {
            warmUpFinished = true;
            log.info("warm up finished");
        }
    }

}
