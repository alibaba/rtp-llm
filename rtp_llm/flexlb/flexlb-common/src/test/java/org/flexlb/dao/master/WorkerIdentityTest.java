package org.flexlb.dao.master;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class WorkerIdentityTest {

    @Test
    void storesEveryWorkerIdentityRepresentation() {
        WorkerIdentity identity = new WorkerIdentity("10.0.0.8", 8080, 1);

        assertEquals("10.0.0.8", identity.getIp());
        assertEquals(8080, identity.getPort());
        assertEquals(1, identity.getEngineIndex());
        assertEquals("10.0.0.8:8080", identity.getPhysicalIpPort());
        assertEquals("10.0.0.8:8080@1", identity.getLogicalIpPort());
    }

    @Test
    void leavesDerivedRepresentationsNullUntilIpIsKnown() {
        WorkerIdentity identity = new WorkerIdentity(null, 8080, 1);

        assertNull(identity.getPhysicalIpPort());
        assertNull(identity.getLogicalIpPort());
    }

    @Test
    void choosesMetricAddressByLogicalWorkerCount() {
        WorkerIdentity identity = new WorkerIdentity("10.0.0.8", 8080, 1);

        assertEquals("10.0.0.8:8080", identity.getMetricIpPort(1));
        assertEquals("10.0.0.8:8080@1", identity.getMetricIpPort(2));
    }

    @Test
    void equalsAndHashCodeCompareAllIdentityFields() {
        WorkerIdentity identity = new WorkerIdentity("10.0.0.8", 8080, 1);

        assertEquals(identity, new WorkerIdentity("10.0.0.8", 8080, 1));
        assertEquals(identity.hashCode(), new WorkerIdentity("10.0.0.8", 8080, 1).hashCode());
        assertNotEquals(identity, new WorkerIdentity("10.0.0.8", 8080, 0));
        assertNotEquals(identity, new WorkerIdentity("10.0.0.9", 8080, 1));
    }
}
