package org.flexlb.balance.strategy;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Verifies frozen selection values and independent protocol responses across worker updates.
 */
class RoleSelectionTest {
    @ParameterizedTest
    @CsvSource({"0,1", "1,3"})
    void projectionPreservesFrozenTopologyAndWireEngineRules(int index, int engines)
            throws Exception {
        var endpoint = mock(PrefillEndpoint.class);
        var worker = mock(WorkerStatus.class);
        when(endpoint.getStatus()).thenReturn(worker);
        when(worker.getRole()).thenReturn(RoleType.PDFUSION);
        when(worker.getDpRank()).thenReturn(7L);
        var topology =
                new WorkerStatus.TopologySnapshot(
                        "selected", "host", 8000, 8001, null, null, index, engines);
        var selection = RoleSelection.prefill(endpoint, topology, 17, 25, null);
        var response = selection.toServerStatus("request");
        assertEquals("host:8000@" + index, response.getLogicalIpPort());
        assertEquals("selected", response.getGroup());
        assertEquals(7, response.getDpRank());
        assertEquals(17, response.getDebugInfo().getHitCacheLen());
        assertEquals(25, selection.computeTokens());
        assertEquals(RoleType.PDFUSION, response.getRole());
        assertEquals("request", response.getRequestId());
        var json = new ObjectMapper().valueToTree(response);
        assertEquals(engines > 1, json.has("engine_index"));
        if (engines > 1) assertEquals(index, json.get("engine_index").intValue());
        assertFalse(json.has("worker_generation"));
        response.setGroup("modified");
        response.getDebugInfo().setHitCacheLen(99);
        var next = selection.toServerStatus("another");
        assertEquals("selected", next.getGroup());
        assertEquals(17, next.getDebugInfo().getHitCacheLen());
        assertEquals("another", next.getRequestId());
    }

    @Test
    void vitUsesFrozenGroupAndTheEndpointsImmutableGeneration() {
        var worker =
                WorkerStatus.createDiscovered(RoleType.VIT, "selected", "host", 8000, 8001, "");
        var endpoint = mock(WorkerEndpoint.class);
        when(endpoint.getStatus()).thenReturn(worker);
        var selection = RoleSelection.auxiliary(endpoint, worker.topologySnapshot());
        worker.lock.lock();
        try {
            worker.updateDiscoveryLabels("new-site", "updated");
        } finally {
            worker.lock.unlock();
        }
        var response = selection.toServerStatus("request");
        assertEquals("selected", response.getGroup());
        assertEquals(worker.getGenerationId(), response.getWorkerGeneration());
        assertEquals("host:8000@0", response.getLogicalIpPort());
    }

    @Test
    void decodeProjectionUsesTheEndpointRankWithoutDuplicatingIt() {
        var endpoint = mock(DecodeEndpoint.class);
        var worker = mock(WorkerStatus.class);
        when(endpoint.getStatus()).thenReturn(worker);
        when(worker.getRole()).thenReturn(RoleType.DECODE);
        when(worker.getDpRank()).thenReturn(7L);
        var selection =
                RoleSelection.decode(
                        endpoint,
                        new WorkerStatus.TopologySnapshot("selected", "host", 8000, 8001, ""),
                        null);
        var response = selection.toServerStatus("request");
        assertEquals(7, response.getDpRank());
        assertNull(response.getDebugInfo());
        when(worker.getDpRank()).thenReturn(9L);
        assertEquals(9, selection.toServerStatus("request").getDpRank());
    }

    @Test
    void invalidEngineIdentityFailsBeforePublishingSelection() {
        var topology =
                new WorkerStatus.TopologySnapshot("group", "host", 8000, 8001, null, null, 2, 1);
        assertThrows(
                IllegalArgumentException.class,
                () -> RoleSelection.decode(mock(DecodeEndpoint.class), topology, null));
    }
}
