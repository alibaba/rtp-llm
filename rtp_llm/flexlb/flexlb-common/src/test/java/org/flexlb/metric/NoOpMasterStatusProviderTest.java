package org.flexlb.metric;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Tests for {@link NoOpMasterStatusProvider}.
 *
 * <p>Verifies that the default (fallback) implementation always
 * returns {@code false} for {@code isMaster()}. This provider is
 * used by non-election roles (e.g. frontend) via
 * {@code @ConditionalOnMissingBean(MasterStatusProvider.class)}.
 *
 * @author saichen.sm
 */
class NoOpMasterStatusProviderTest {

    private NoOpMasterStatusProvider provider;

    @BeforeEach
    void setUp() {
        provider = new NoOpMasterStatusProvider();
    }

    @Test
    @DisplayName("isMaster always returns false")
    void isMasterAlwaysReturnsFalse() {
        assertFalse(provider.isMaster());
    }

    @Test
    @DisplayName("isMaster returns false on repeated calls")
    void isMasterReturnsFalseOnRepeatedCalls() {
        for (int i = 0; i < 10; i++) {
            assertFalse(provider.isMaster(), "Call #" + i + " should return false");
        }
    }
}
