package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.mockito.Mockito;

/**
 * Creates cancellation clients and exact route-publication fixtures for scheduler tests.
 */
public final class DeliveryTestSupport {
    private DeliveryTestSupport() {}

    public static void setField(Object target, String name, Object value) {
        org.springframework.test.util.ReflectionTestUtils.setField(target, name, value);
    }

    public static Object getField(Object target, String name) {
        return org.springframework.test.util.ReflectionTestUtils.getField(target, name);
    }

    public static EngineCancellation route(DeliveryMetricsReporter reporter) {
        return new EngineCancellation(Mockito.mock(EngineGrpcClient.class));
    }

    public static ConfigService service(FlexlbConfig config) {
        var service = Mockito.mock(ConfigService.class);
        Mockito.when(service.loadBalanceConfig()).thenReturn(config);
        return service;
    }

    static EngineCancellation cancellations(EngineCancellation cancellation) {
        return cancellation;
    }
}
