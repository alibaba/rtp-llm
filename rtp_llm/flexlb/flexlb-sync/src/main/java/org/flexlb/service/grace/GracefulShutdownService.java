package org.flexlb.service.grace;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.listener.ShutdownListener;
import org.springframework.stereotype.Component;
import sun.misc.Signal;
import sun.misc.SignalHandler;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * 服务优雅上下线配置
 */
@Slf4j
@Component
public class GracefulShutdownService implements SignalHandler {

    private static final List<ShutdownListener> SHUTDOWN_LISTENERS = new CopyOnWriteArrayList<>();

    public GracefulShutdownService() {
        Signal sig = new Signal("USR2");
        Signal.handle(sig, this);
    }

    @Override
    public void handle(Signal signal) {
        log.info("receive signal: {}", signal.getName());
        for (ShutdownListener listener : SHUTDOWN_LISTENERS) {
            listener.beforeShutdown();
        }
    }

    public static void addShutdownListener(ShutdownListener listener) {
        SHUTDOWN_LISTENERS.add(listener);
    }

}
