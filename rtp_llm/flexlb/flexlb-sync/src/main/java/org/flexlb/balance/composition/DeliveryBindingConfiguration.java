package org.flexlb.balance.composition;

import org.flexlb.balance.delivery.BatchSubmissionPort;
import org.flexlb.balance.delivery.DeliveryTelemetry;
import org.flexlb.balance.endpoint.EndpointRequestRuntime;
import org.flexlb.balance.endpoint.PrefillDeliveryStrategyBinding;
import org.flexlb.balance.scheduler.BatchIdGenerator;
import org.flexlb.balance.scheduler.BatchPrefillAdmission;
import org.flexlb.balance.scheduler.DeliveryTelemetryAdapter;
import org.flexlb.balance.scheduler.RoutePrefillAdmission;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.Objects;

/** Selects exactly one delivery binding from the canonical startup config. */
@Configuration(proxyBeanMethods = false)
public class DeliveryBindingConfiguration {

    @Bean
    public PrefillDeliveryStrategyBinding activePrefillDeliveryBinding(
            ConfigService configService,
            EndpointRequestRuntime requestRuntime,
            BatchSubmissionPort batchSubmission,
            BatchSchedulerReporter reporter,
            Environment environment) {
        DispatcherConfig dispatcher = Objects.requireNonNull(
                configService.loadBalanceConfig().getDispatcher(),
                "dispatcher");
        DeliveryTelemetry telemetry = new DeliveryTelemetryAdapter(reporter);
        return switch (dispatcher) {
            case BatchDispatcherConfig ignored -> {
                BatchIdGenerator ids = new BatchIdGenerator(
                        detectLocalIp(), detectPort(environment));
                yield new BatchDeliveryBinding(
                        batchSubmission,
                        new BatchPrefillAdmission(ids::nextBatchId),
                        requestRuntime,
                        telemetry);
            }
            case NonBatchDispatcherConfig ignored ->
                    new RouteDeliveryBinding(
                            new RoutePrefillAdmission(),
                            requestRuntime,
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
