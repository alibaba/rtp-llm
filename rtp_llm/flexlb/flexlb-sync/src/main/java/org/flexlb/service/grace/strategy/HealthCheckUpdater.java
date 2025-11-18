package org.flexlb.service.grace.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.listener.ShutdownListener;
import org.flexlb.service.grace.GracefulShutdownService;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class HealthCheckUpdater implements ShutdownListener {

    public static boolean isShutDownSignalReceived;

    public HealthCheckUpdater() {
        GracefulShutdownService.addShutdownListener(this);
    }

    @Override
    public void beforeShutdown() {
        isShutDownSignalReceived = true;
    }
}
