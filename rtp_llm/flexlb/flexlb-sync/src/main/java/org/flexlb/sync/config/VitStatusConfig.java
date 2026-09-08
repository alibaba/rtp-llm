package org.flexlb.sync.config;

import org.flexlb.util.EnvUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class VitStatusConfig {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final AtomicBoolean timeoutWindowWarningLogged = new AtomicBoolean();

    public static final long SYNC_REQUEST_TIMEOUT_MS =
            EnvUtils.readPositiveLong("VIT_SYNC_REQUEST_TIMEOUT_MS", 2000L);
    public static final long WORKER_TIMEOUT_US =
            EnvUtils.readPositiveLong("VIT_WORKER_TIMEOUT_US", 5_000_000L);
    public static final boolean RETAIN_ALIVE_ON_TIMEOUT =
            EnvUtils.readBoolean("VIT_RETAIN_ALIVE_ON_TIMEOUT", true);

    static {
        logger.info("VIT worker status config: timeoutMs={}, workerTimeoutUs={}, retainAliveOnTimeout={}",
                SYNC_REQUEST_TIMEOUT_MS, WORKER_TIMEOUT_US, RETAIN_ALIVE_ON_TIMEOUT);
    }

    private VitStatusConfig() {
    }

    public static void warnIfRetentionWindowAtRisk(long effectiveRpcTimeoutMs) {
        long effectiveRpcTimeoutUs = TimeUnit.MILLISECONDS.toMicros(effectiveRpcTimeoutMs);
        long warningThresholdUs = WORKER_TIMEOUT_US / 2 + WORKER_TIMEOUT_US % 2;
        if (effectiveRpcTimeoutUs >= warningThresholdUs
                && timeoutWindowWarningLogged.compareAndSet(false, true)) {
            logger.warn("Effective VIT status RPC timeout {}ms consumes at least half of the worker expiration "
                            + "window {}us; the endpoint may expire before the next successful check "
                            + "(retainAliveOnTimeout={})",
                    effectiveRpcTimeoutMs, WORKER_TIMEOUT_US, RETAIN_ALIVE_ON_TIMEOUT);
        }
    }
}
