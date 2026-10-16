package org.flexlb.httpserver;

import org.junit.jupiter.api.Test;
import org.springframework.mock.env.MockEnvironment;
import static org.junit.jupiter.api.Assertions.*;

class GrpcServerBindingTest {
    @Test void absentOrBlankAddressPreservesDefaultBuilder() {
        assertNull(GrpcServerBinding.configuredAddress(new MockEnvironment(), 7003));
        assertNull(GrpcServerBinding.configuredAddress(new MockEnvironment().withProperty("server.address", " "), 7003));
        assertNotNull(GrpcServerBinding.builder(new MockEnvironment(), 7003));
    }

    @Test void explicitAddressRetainsGrpcPortAndSeparatesLocalMasters() {
        var address = GrpcServerBinding.configuredAddress(
                new MockEnvironment().withProperty("server.address", "127.0.0.2"), 7003);
        assertEquals("127.0.0.2", address.getAddress().getHostAddress());
        assertEquals(7003, address.getPort());
        assertNotNull(GrpcServerBinding.builder(new MockEnvironment().withProperty("server.address", "127.0.0.2"), 7003));
    }
}
