package org.flexlb.balance.schedulev2;

import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigService;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class RequestStatusTimeoutTest {
    @Test
    void statusTimeoutDoesNotChangeWithPollInterval() {
        var config = SchedulingTestConfig.newConfig();
        config.getRequestLifecycle().getRequest().setStatusUpdateTimeoutMs(1_000L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        config.getWorkerRegistry().getHealth().setStatusPollIntervalMs(20L);
        assertEquals(1_000L, RequestStatusTimeout.timeoutMs(service));
        config.getWorkerRegistry().getHealth().setStatusPollIntervalMs(200L);
        assertEquals(1_000L, RequestStatusTimeout.timeoutMs(service));
    }
}
