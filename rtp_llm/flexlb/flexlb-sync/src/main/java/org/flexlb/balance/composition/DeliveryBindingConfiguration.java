package org.flexlb.balance.composition;

import org.flexlb.balance.delivery.BatchDeliveryStrategy;
import org.flexlb.balance.delivery.BatchSubmissionPort;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.delivery.RouteDeliveryStrategy;
import org.flexlb.balance.scheduler.BatchIdGenerator;
import org.flexlb.balance.scheduler.BatchPrefillAdmission;
import org.flexlb.balance.scheduler.RoutePrefillAdmission;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.Objects;

/** Selects exactly one delivery strategy from the canonical startup config. */
@Configuration(proxyBeanMethods = false)
public class DeliveryBindingConfiguration {

    @Bean
    public DeliveryStrategy activePrefillDeliveryStrategy(
            ConfigService configService,
            SlotDeliveryPort slotDelivery,
            BatchSubmissionPort batchSubmission,
            BatchSchedulerReporter reporter,
            Environment environment) {
        DispatcherConfig dispatcher = Objects.requireNonNull(
                configService.loadBalanceConfig().getDispatcher(),
                "dispatcher");
        DeliveryMetrics telemetry = new DeliveryMetrics(reporter);
        return switch (dispatcher.getType()) {
            case BATCH -> {
                BatchIdGenerator ids = new BatchIdGenerator(
                        detectLocalIp(), detectPort(environment));
                yield new BatchDeliveryStrategy(
                        batchSubmission,
                        new BatchPrefillAdmission(ids::nextBatchId),
                        slotDelivery,
                        telemetry);
            }
            case NON_BATCH ->
                    new RouteDeliveryStrategy(
                            new RoutePrefillAdmission(),
                            slotDelivery,
                            telemetry);
        };
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
