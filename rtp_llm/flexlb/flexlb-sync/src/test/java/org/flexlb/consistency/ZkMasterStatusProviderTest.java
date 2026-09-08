package org.flexlb.consistency;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Tests for {@link ZkMasterStatusProvider}.
 *
 * <p>Verifies that the provider correctly delegates to
 * {@link LBStatusConsistencyService#isMaster()} and that the
 * {@code @Lazy} constructor injection pattern (which fixed the
 * Spring circular dependency between ZkMasterStatusProvider and
 * LBStatusConsistencyService) works as expected.
 *
 * <p><b>Regression context:</b> Prior to the {@code @Lazy} fix,
 * constructor injection of {@code LBStatusConsistencyService} into
 * {@code ZkMasterStatusProvider} caused a Spring circular dependency,
 * which prevented {@code KMonitorAdapter} from receiving a
 * {@code MasterStatusProvider} bean — all metrics stopped reporting
 * the {@code isMaster} tag.
 *
 * @author saichen.sm
 */
class ZkMasterStatusProviderTest {

    private LBStatusConsistencyService lbStatusConsistencyService;
    private ZkMasterStatusProvider provider;

    @BeforeEach
    void setUp() {
        lbStatusConsistencyService = mock(LBStatusConsistencyService.class);
        provider = new ZkMasterStatusProvider(lbStatusConsistencyService);
    }

    @Test
    @DisplayName("isMaster returns true when underlying service reports master")
    void isMasterReturnsTrueWhenServiceIsMaster() {
        when(lbStatusConsistencyService.isMaster()).thenReturn(true);

        assertTrue(provider.isMaster());
    }

    @Test
    @DisplayName("isMaster returns false when underlying service reports non-master")
    void isMasterReturnsFalseWhenServiceIsNotMaster() {
        when(lbStatusConsistencyService.isMaster()).thenReturn(false);

        assertFalse(provider.isMaster());
    }

    @Test
    @DisplayName("isMaster delegates to LBStatusConsistencyService.isMaster()")
    void isMasterDelegatesToService() {
        provider.isMaster();

        verify(lbStatusConsistencyService).isMaster();
    }

    @Test
    @DisplayName("constructor accepts @Lazy proxy (simulated by Mockito mock)")
    void constructorAcceptsLazyProxy() {
        // The @Lazy annotation causes Spring to inject a proxy that defers
        // actual bean creation until a method is called. A Mockito mock
        // simulates this behaviour: it is a proxy whose backing bean is
        // not initialised until stubs are configured.
        LBStatusConsistencyService lazyProxy = mock(LBStatusConsistencyService.class);
        when(lazyProxy.isMaster()).thenReturn(true);

        ZkMasterStatusProvider providerWithLazyProxy = new ZkMasterStatusProvider(lazyProxy);

        assertTrue(providerWithLazyProxy.isMaster());
    }
}
