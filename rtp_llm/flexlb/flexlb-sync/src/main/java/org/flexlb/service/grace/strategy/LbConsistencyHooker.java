package org.flexlb.service.grace.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.listener.AppOnlineHooker;
import org.flexlb.listener.AppShutDownHooker;
import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.flexlb.service.grace.GracefulOnlineService;
import org.flexlb.service.grace.GracefulShutdownService;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class LbConsistencyHooker implements AppShutDownHooker, AppOnlineHooker {

    private final LBStatusConsistencyService lbStatusConsistencyService;
    private final GracefulLifecycleReporter lifecycleReporter;

    public LbConsistencyHooker(LBStatusConsistencyService lbStatusConsistencyService, GracefulLifecycleReporter lifecycleReporter) {
        this.lbStatusConsistencyService = lbStatusConsistencyService;
        this.lifecycleReporter = lifecycleReporter;
        GracefulShutdownService.addShutdownListener(this);
        GracefulOnlineService.addOnlineListener(this);
    }

    @Override
    public void beforeShutdown() {
        try {
            long startTime = System.currentTimeMillis();
            lbStatusConsistencyService.offline();
            lifecycleReporter.reportZkNodeOffline(System.currentTimeMillis() - startTime);
        } catch (Throwable throwable) {
            Logger.error("handle beforeShutdown error", throwable);
        }
    }

    @Override
    public void afterStartUp() {
        try {
            long startTime = System.currentTimeMillis();
            lbStatusConsistencyService.start();
            lifecycleReporter.reportZkNodeOnline(System.currentTimeMillis() - startTime);
        } catch (Exception e) {
            Logger.error("handle afterStartUp error", e);
        }
    }

    @Override
    public int priority() {
        return 3;
    }
}