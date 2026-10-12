package org.flexlb.consistency;

import org.junit.jupiter.api.Test;
import org.springframework.mock.env.MockEnvironment;
import static org.junit.jupiter.api.Assertions.*;

class MasterHostTest {
    @Test void explicitIdentitySeparatesSameHostMasters() {
        assertEquals("127.0.0.1", MasterHost.resolve(new MockEnvironment().withProperty("FLEXLB_ADVERTISED_IP", "127.0.0.1")));
        assertEquals("127.0.0.2", MasterHost.resolve(new MockEnvironment().withProperty("FLEXLB_ADVERTISED_IP", "127.0.0.2")));
    }
    @Test void wildcardIsNotAnElectionIdentity() {
        assertThrows(IllegalArgumentException.class, () -> MasterHost.resolve(new MockEnvironment().withProperty("flexlb.advertised-ip", "0.0.0.0")));
    }
}
