package org.flexlb.dao.master;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
        assertEquals("10.0.0.8@1", identity.getIpIndex());
    }

    @Test
    void leavesDerivedRepresentationsNullUntilIpIsKnown() {
        WorkerIdentity identity = new WorkerIdentity(null, 8080, 1);

        assertNull(identity.getPhysicalIpPort());
        assertNull(identity.getLogicalIpPort());
        assertNull(identity.getIpIndex());
    }

    @Test
    void workerHostExposesItsStoredIdentityRepresentations() {
        WorkerHost host = new WorkerHost(
                "10.0.0.8", 8080, 8081, 8085, 18003,
                "site-a", "group-a", "deployment-a", 1, 2);

        assertEquals("10.0.0.8", host.getIp());
        assertEquals(8080, host.getPort());
        assertEquals(1, host.getEngineIndex());
        assertEquals("10.0.0.8:8080", host.getPhysicalIpPort());
        assertEquals("10.0.0.8:8080@1", host.getLogicalIpPort());
        assertEquals("10.0.0.8@1", host.getIpIndex());
    }
}
