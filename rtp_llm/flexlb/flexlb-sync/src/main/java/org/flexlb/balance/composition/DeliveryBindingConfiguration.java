package org.flexlb.balance.composition;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.scheduler.BatchDeliveryStrategy;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;

import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongSupplier;

/** Selects exactly one delivery strategy from the canonical startup config. */
@Configuration(proxyBeanMethods = false)
public class DeliveryBindingConfiguration {

    private static final long BATCH_EPOCH_SECONDS = 1577836800L;
    private static final long BATCH_SEQUENCE_MASK = (1L << 21) - 1;
    private static final AtomicLong BATCH_SEQUENCE = new AtomicLong();

    @Bean
    public DeliveryStrategy activePrefillDeliveryStrategy(
            ConfigService configService,
            RequestRegistry requests,
            DefaultBatchDispatcher batchSubmission,
            BatchSchedulerReporter reporter,
            Environment environment) {
        DispatcherConfig dispatcher = Objects.requireNonNull(
                configService.loadBalanceConfig().getDispatcher(),
                "dispatcher");
        DeliveryMetrics telemetry = new DeliveryMetrics(reporter);
        return switch (dispatcher.getType()) {
            case BATCH -> {
                LongSupplier ids = batchIds(
                        detectLocalIp(), detectPort(environment));
                yield new BatchDeliveryStrategy(
                        batchSubmission::tryPrepareSubmission,
                        ids,
                        requests,
                        telemetry);
            }
            case NON_BATCH ->
                    new RouteDeliveryStrategy(
                            requests,
                            telemetry);
        };
    }

    /** Snowflake-compatible 31-bit timestamp, 12-bit master, 21-bit sequence. */
    private static LongSupplier batchIds(String localIp, int port) {
        long masterId = computeMasterId(
                localIp, port, ProcessHandle.current().pid(),
                ManagementFactory.getRuntimeMXBean().getStartTime());
        return () -> {
            long timestamp = System.currentTimeMillis() / 1000L
                    - BATCH_EPOCH_SECONDS;
            if (timestamp < 0L || (timestamp >>> 31) != 0L) {
                throw new IllegalStateException(
                        timestamp < 0L
                                ? "system clock is before the batch ID epoch"
                                : "batch ID timestamp overflow");
            }
            return (timestamp << 33)
                    | (masterId << 21)
                    | (BATCH_SEQUENCE.getAndIncrement()
                    & BATCH_SEQUENCE_MASK);
        };
    }

    private static long computeMasterId(
            String localIp, int port, long processId,
            long processStartTimeMs) {
        try {
            String input = localIp + ":" + port + ":" + processId
                    + ":" + processStartTimeMs;
            byte[] hash = MessageDigest.getInstance("SHA-256")
                    .digest(input.getBytes(StandardCharsets.UTF_8));
            return (hash[hash.length - 1] & 0xFFL)
                    | ((hash[hash.length - 2] & 0x0FL) << 8);
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 not available", impossible);
        }
    }

    private static String detectLocalIp() {
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException failure) {
            Logger.warn(
                    "Failed to detect local IP for batch ids; using loopback",
                    failure);
            return "127.0.0.1";
        }
    }

    private static int detectPort(Environment environment) {
        String configured = environment == null
                ? null : environment.getProperty("server.port");
        if (configured == null) {
            configured = System.getProperty("server.port", "7001");
        }
        try {
            return Integer.parseInt(configured);
        } catch (NumberFormatException ignored) {
            return 7001;
        }
    }
}
