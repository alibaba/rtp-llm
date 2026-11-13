package org.flexlb.config;

import com.google.common.net.InetAddresses;
import org.junit.jupiter.api.Test;

import java.net.Inet4Address;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks that the local Mock override accepts alternate loopback IPs and rejects external IPs.
 */
class MasterLoopbackAddressTest {
    @Test
    void acceptsSecondaryLoopbackAddress() {
        var address = (Inet4Address) InetAddresses.forString("127.0.0.2");
        assertEquals(address, new MasterLoopbackAddress(address).address());
    }

    @Test
    void rejectsNonLoopbackAddress() {
        var address = (Inet4Address) InetAddresses.forString("192.0.2.1");
        assertThrows(IllegalArgumentException.class, () -> new MasterLoopbackAddress(address));
    }
}
