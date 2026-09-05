package org.flexlb.dao.master;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class WorkerHostTest {

    @Test
    void exposesItsStoredIdentityRepresentations() {
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

    @Test
    void exposesMultiEnginePortsAndTopology() {
        WorkerHost host = new WorkerHost(
                "10.0.0.8", 8080, 8081, 8085, 18003,
                "site-a", "group-a", "deployment-a", 1, 2);

        assertEquals(18003, host.getWorkerStatusPort());
        assertEquals(2, host.getMultiEngineNum());
        assertEquals("site-a", host.getSite());
        assertEquals("group-a", host.getGroup());
        assertEquals("deployment-a", host.getDeploymentName());
    }
}
