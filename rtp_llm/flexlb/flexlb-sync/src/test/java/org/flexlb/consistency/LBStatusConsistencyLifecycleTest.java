package org.flexlb.consistency;

import org.flexlb.domain.consistency.LBConsistencyConfig;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.mock.env.MockEnvironment;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class LBStatusConsistencyLifecycleTest {

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void contextShutdownClosesOnlyEnabledElection(boolean enabled) {
        var election = mock(ZookeeperMasterElectService.class);
        var config = new LBConsistencyConfig();
        config.setNeedConsistency(enabled);
        when(election.getLbConsistencyConfig()).thenReturn(config);
        try (var context = new AnnotationConfigApplicationContext()) {
            context.registerBean(LBStatusConsistencyService.class,
                    () -> new LBStatusConsistencyService(election, new MockEnvironment()));
            context.refresh();
            verify(election, times(0)).destroy();
        }
        verify(election, times(enabled ? 1 : 0)).destroy();
    }
}
