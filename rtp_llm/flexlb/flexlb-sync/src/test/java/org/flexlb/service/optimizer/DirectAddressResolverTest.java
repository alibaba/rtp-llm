package org.flexlb.service.optimizer;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DirectAddressResolverTest {

    @Test
    void should_return_configured_address() {
        DirectAddressResolver resolver = new DirectAddressResolver("10.0.0.1:8082");

        List<String> addresses = resolver.getAddresses();

        assertEquals(1, addresses.size());
        assertEquals("10.0.0.1:8082", addresses.get(0));
    }

    @Test
    void should_shutdown_without_error() {
        DirectAddressResolver resolver = new DirectAddressResolver("10.0.0.1:8082");

        resolver.shutdown();

        assertEquals(List.of("10.0.0.1:8082"), resolver.getAddresses());
    }
}
