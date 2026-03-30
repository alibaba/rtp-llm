package org.flexlb.service.grace.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.listener.AppShutDownHooker;
import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.flexlb.service.grace.GracefulShutdownService;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class HealthCheckHooker implements AppShutDownHooker {

    public static boolean isShutDownSignalReceived;
    private final GracefulLifecycleReporter lifecycleReporter;

    public HealthCheckHooker(GracefulLifecycleReporter lifecycleReporter) {
        this.lifecycleReporter = lifecycleReporter;
        GracefulShutdownService.addShutdownListener(this);
    }

    @Override
    public void beforeShutdown() {
        log.info("set isShutDownSignalReceived to true");
        isShutDownSignalReceived = true;
        lifecycleReporter.reportHealthCheckOffline(0);
    }
}
