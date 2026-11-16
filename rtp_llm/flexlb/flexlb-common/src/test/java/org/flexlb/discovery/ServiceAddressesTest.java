package org.flexlb.discovery;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;

/**
 * Verifies that service bootstrap addresses preserve the configured port source.
 */
class ServiceAddressesTest {

    @Test
    void resolvesFixedHostAndPort() {
        assertEquals(List.of("127.0.0.1:6381"), ServiceAddresses.resolve("127.0.0.1:6381", null));
        assertEquals(List.of("[::1]:6381"), ServiceAddresses.resolve("[::1]:6381", null));
        assertThrows(IllegalArgumentException.class, () -> ServiceAddresses.resolve("127.0.0.1", null));
    }

    @Test
    void resolvesDiscoveryWithoutEnginePortConversion() {
        var discovery = new LocalServiceDiscovery(Map.of(
                "meta", List.of("127.0.0.1:6381", "127.0.0.2:6382")));
        assertEquals(List.of("127.0.0.1:6381", "127.0.0.2:6382"),
                ServiceAddresses.resolve("meta", discovery));
        assertEquals("127.0.0.1:7381", ServiceAddresses.resolveFirst("meta", 7381, discovery));
        assertEquals("missing:7381", ServiceAddresses.resolveFirst("missing", 7381, discovery));
        assertEquals(List.of(), ServiceAddresses.resolve("missing", discovery));
    }

    @Test
    void fixedMasterAddressDoesNotConsultDiscovery() {
        var discovery = mock(ServiceDiscovery.class);
        assertEquals("127.0.0.1:7381", ServiceAddresses.resolveFirst("127.0.0.1", 7381, discovery));
        assertEquals("[::1]:7381", ServiceAddresses.resolveFirst("::1", 7381, discovery));
        verifyNoInteractions(discovery);
    }
}
