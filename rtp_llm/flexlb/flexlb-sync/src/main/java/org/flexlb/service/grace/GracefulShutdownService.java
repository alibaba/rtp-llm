package org.flexlb.service.grace;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.listener.AppShutDownHooker;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

@Slf4j
@Component
public class GracefulShutdownService {

    private static final List<AppShutDownHooker> SHUTDOWN_LISTENERS = new CopyOnWriteArrayList<>();

    public static void addShutdownListener(AppShutDownHooker listener) {
        SHUTDOWN_LISTENERS.add(listener);
    }

    public void offline() {
        for (AppShutDownHooker listener : SHUTDOWN_LISTENERS) {
            listener.beforeShutdown();
        }
    }
}
