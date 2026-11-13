package org.flexlb.consistency;

import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ZookeeperConsistencyConfig;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.mock.env.MockEnvironment;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies that Spring shutdown closes only an enabled election.
 */
class LBStatusConsistencyLifecycleTest {

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void contextShutdownClosesOnlyEnabledElection(boolean enabled) {
        var election = mock(ZookeeperMasterElectService.class);
        var config = new FlexlbConfig();
        if (enabled) {
            config.setConsistency(new ZookeeperConsistencyConfig());
        }
        var configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        var identity = mock(DeploymentIdentity.class);
        try (var context = new AnnotationConfigApplicationContext()) {
            context.registerBean(LBStatusConsistencyService.class,
                    () -> new LBStatusConsistencyService(election, new MockEnvironment(), configService, identity));
            context.refresh();
            verify(election, times(0)).destroy();
        }
        verify(election, times(enabled ? 1 : 0)).destroy();
    }
}
