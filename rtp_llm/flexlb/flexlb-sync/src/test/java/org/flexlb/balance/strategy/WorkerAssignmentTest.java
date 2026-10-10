package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

class WorkerAssignmentTest {
    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE", "VIT"})
    void failedMetadataValidationReleasesTheConsumedPin(RoleType role) {
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        ServerStatus response = new ServerStatus();
        response.setRole(role);
        response.setSuccess(false);

        assertThrows(IllegalArgumentException.class, () -> {
            switch (role) {
                case PREFILL -> WorkerAssignment.prefill(pin, response, 10L, 0L);
                case DECODE -> WorkerAssignment.decode(pin, response, 0L);
                case VIT -> WorkerAssignment.stateless(pin, response);
                default -> throw new AssertionError(role);
            }
        });

        verify(pin).close();
        verifyNoMoreInteractions(pin);
    }

    @Test
    void assignmentConsumesAPinOnceAndRetainsFactsAfterThePinCloses() {
        var status = org.flexlb.dao.master.WorkerStatus.createDiscovered(
                RoleType.VIT, null, "127.0.0.1", 8080, 8081, null);
        WorkerEndpoint endpoint = new WorkerEndpoint(status);
        WorkerEndpoint.GenerationPin pin = org.mockito.Mockito.spy(endpoint.tryPinGeneration());
        ServerStatus response = new ServerStatus();
        response.setSuccess(true);
        response.setRole(RoleType.VIT);
        response.setServerIp("127.0.0.1");
        response.setHttpPort(8080);
        WorkerAssignment selected = WorkerAssignment.stateless(pin, response);
        try {
            selected.assignToRequest();
            assertThrows(IllegalStateException.class, selected::assignToRequest);
            org.junit.jupiter.api.Assertions.assertSame(pin, selected.generationPin());
            org.mockito.Mockito.verify(pin, org.mockito.Mockito.never()).close();
            selected.close();
            selected.close();
            assertEquals(false, pin.isOpen());
            org.junit.jupiter.api.Assertions.assertSame(endpoint, selected.endpoint());
            org.junit.jupiter.api.Assertions.assertEquals(response, selected.serverStatus());
            assertEquals(0L, selected.placementVersion());
            assertThrows(IllegalStateException.class, selected::generationPin);
            assertThrows(IllegalStateException.class, selected::assignToRequest);
        } finally {
            selected.close();
            endpoint.close();
            endpoint.awaitRetirement();
        }
    }

    @Test
    void metadataIsFrozenBeforeCallerOrResponseMutations() {
        var worker = org.flexlb.dao.master.WorkerStatus.createDiscovered(
                RoleType.VIT, null, "127.0.0.1", 8080, 8081, null);
        WorkerEndpoint endpoint = new WorkerEndpoint(worker);
        var metadata = new ServerStatus();
        metadata.setSuccess(true);
        metadata.setRole(RoleType.VIT);
        metadata.setServerIp(endpoint.getIp());
        metadata.setHttpPort(endpoint.getHttpPort());
        metadata.setRequestId(42L);
        metadata.setGroup("original");
        var debug = new org.flexlb.dao.loadbalance.DebugInfo();
        debug.setHitCacheLen(17L);
        metadata.setDebugInfo(debug);
        try (var assignment = WorkerAssignment.stateless(endpoint.tryPinGeneration(), metadata)) {
            metadata.setRequestId(99L);
            metadata.setGroup("changed");
            debug.setHitCacheLen(999L);
            var exported = assignment.serverStatus();
            exported.setGroup("response mutation");
            exported.getDebugInfo().setHitCacheLen(1L);
            assertEquals(42L, assignment.requestId());
            assertEquals("original", assignment.group());
            assertEquals(17L, assignment.hitCache());
            assertEquals("original", assignment.serverStatus().getGroup());
            assignment.close();
            assertEquals(42L, assignment.requestId());
            assertEquals(17L, assignment.hitCache());
        } finally {
            endpoint.close();
            endpoint.awaitRetirement();
        }
    }

    @Test
    void aClosedGenerationCannotBeAssignedEvenWhenNoRequestConsumedIt() {
        var worker = org.flexlb.dao.master.WorkerStatus.createDiscovered(
                RoleType.VIT, null, "127.0.0.1", 8080, 8081, null);
        WorkerEndpoint endpoint = new WorkerEndpoint(worker);
        var metadata = new ServerStatus();
        metadata.setSuccess(true);
        metadata.setRole(RoleType.VIT);
        metadata.setServerIp(endpoint.getIp());
        metadata.setHttpPort(endpoint.getHttpPort());
        var pin = endpoint.tryPinGeneration();
        try (var assignment = WorkerAssignment.stateless(pin, metadata)) {
            pin.close();
            assertThrows(IllegalStateException.class, assignment::assignToRequest);
            assertThrows(IllegalStateException.class, assignment::generationPin);
            assertEquals(RoleType.VIT, assignment.role());
        } finally {
            endpoint.close();
            endpoint.awaitRetirement();
        }
    }

    @Test
    void failedConstructionPreservesItsCauseWhenPinReleaseAlsoFails() {
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        IllegalStateException cleanup = new IllegalStateException("pin release failed");
        doThrow(cleanup).when(pin).close();
        ServerStatus response = new ServerStatus();
        response.setSuccess(false);

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> WorkerAssignment.stateless(pin, response));

        assertEquals("WorkerAssignment requires successful response metadata", failure.getMessage());
        assertEquals(List.of(cleanup), List.of(failure.getSuppressed()));
        verify(pin).close();
        verifyNoMoreInteractions(pin);
    }

    @ParameterizedTest
    @EnumSource(value = RoleType.class, names = {"PREFILL", "DECODE"})
    void factoryValidationKeepsItsCauseWhenPinReleaseAlsoFails(RoleType role) {
        WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
        IllegalStateException cleanup = new IllegalStateException("pin release failed");
        doThrow(cleanup).when(pin).close();

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class, () -> {
            if (role == RoleType.PREFILL) {
                WorkerAssignment.prefill(pin, null, -1L, 0L);
            } else {
                WorkerAssignment.decode(pin, null, 0L);
            }
        });

        assertEquals(role == RoleType.PREFILL ? "Prefill work must be non-negative"
                : "Decode selection requires Decode metadata", failure.getMessage());
        assertEquals(List.of(cleanup), List.of(failure.getSuppressed()));
        verify(pin).close();
        verifyNoMoreInteractions(pin);
    }
}
