package org.flexlb.constraint;

import org.flexlb.constraint.ConstraintTreeModels.ArtifactMetadata;
import org.flexlb.constraint.ConstraintTreeModels.PublicationResult;
import org.flexlb.constraint.ConstraintTreeModels.SerializedArtifact;
import org.flexlb.constraint.ConstraintTreeModels.WorkerUpdateResponse;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.address.WorkerAddressService;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class WhaleConstraintTreePublisherTest {

    private final WorkerAddressService addresses = mock(WorkerAddressService.class);
    private final GeneralHttpNettyService http = mock(GeneralHttpNettyService.class);
    private final WhaleConstraintTreePublisher publisher = new WhaleConstraintTreePublisher(
            addresses, http, 2, Duration.ofSeconds(2));

    @AfterEach
    void tearDown() {
        publisher.destroy();
    }

    @Test
    void publishesToCppHttpPortsAndDeduplicatesWorkersAcrossRoles() {
        WorkerHost first = new WorkerHost("10.0.0.1", 8000, 8001, 8005, "hz", "default");
        WorkerHost second = new WorkerHost("10.0.0.2", 9000, 9001, 9005, "sh", "default");
        when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of(first));
        when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of(first, second));
        when(http.requestRawJson(any(byte[].class), any(URI.class),
                eq(WhaleConstraintTreePublisher.UPDATE_PATH), eq(WorkerUpdateResponse.class)))
                .thenReturn(Mono.just(new WorkerUpdateResponse("accepted", 0, 7, "queued", false, 0)));

        SerializedArtifact artifact = artifact();
        PublicationResult result = publisher.publish(artifact);

        assertEquals(2, result.targetWorkerCount());
        assertEquals(2, result.publishedWorkerCount());
        verify(http).requestRawJson(eq(artifact.payload()), eq(URI.create("http://10.0.0.1:8005")),
                eq(WhaleConstraintTreePublisher.UPDATE_PATH), eq(WorkerUpdateResponse.class));
        verify(http).requestRawJson(eq(artifact.payload()), eq(URI.create("http://10.0.0.2:9005")),
                eq(WhaleConstraintTreePublisher.UPDATE_PATH), eq(WorkerUpdateResponse.class));
    }

    @Test
    void reportsNoTargetsWithoutFailingTheBuild() {
        when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of());
        when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of());

        PublicationResult result = publisher.publish(artifact());

        assertEquals(0, result.targetWorkerCount());
        assertEquals(0, result.publishedWorkerCount());
    }

    @Test
    void statusProbeSkipsPayloadWhenWorkerAlreadyHasVersion() {
        WorkerHost worker = new WorkerHost("10.0.0.1", 8000, 8001, 8005, "hz", "default");
        when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of(worker));
        when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of());
        when(http.get(URI.create("http://10.0.0.1:8005"), WhaleConstraintTreePublisher.STATUS_PATH,
                WorkerUpdateResponse.class))
                .thenReturn(Mono.just(new WorkerUpdateResponse("ready", 7, 7, "ready", true, 4)));

        PublicationResult result = publisher.publish(artifact());

        assertEquals(1, result.publishedWorkerCount());
        verify(http, never()).requestRawJson(any(byte[].class), any(URI.class),
                eq(WhaleConstraintTreePublisher.UPDATE_PATH), eq(WorkerUpdateResponse.class));
    }

    @Test
    void newerWorkerVersionIsReportedAsConflictWithoutOverwritingIt() {
        WorkerHost worker = new WorkerHost("10.0.0.1", 8000, 8001, 8005, "hz", "default");
        when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of(worker));
        when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of());
        when(http.get(URI.create("http://10.0.0.1:8005"), WhaleConstraintTreePublisher.STATUS_PATH,
                WorkerUpdateResponse.class))
                .thenReturn(Mono.just(new WorkerUpdateResponse("ready", 8, 8, "ready", true, 4)));

        PublicationResult result = publisher.publish(artifact());

        assertEquals(0, result.publishedWorkerCount());
        assertEquals(8, result.workers().get(0).version());
        verify(http, never()).requestRawJson(any(byte[].class), any(URI.class),
                eq(WhaleConstraintTreePublisher.UPDATE_PATH), eq(WorkerUpdateResponse.class));
    }

    @Test
    void statusProbeDoesNotResendVersionThatWorkerIsAlreadyLoading() {
        WorkerHost worker = new WorkerHost("10.0.0.1", 8000, 8001, 8005, "hz", "default");
        when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of(worker));
        when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of());
        when(http.get(URI.create("http://10.0.0.1:8005"), WhaleConstraintTreePublisher.STATUS_PATH,
                WorkerUpdateResponse.class))
                .thenReturn(Mono.just(new WorkerUpdateResponse("loading", 6, 7, "loading", true, 3)));

        PublicationResult result = publisher.publish(artifact());

        assertEquals(1, result.publishedWorkerCount());
        verify(http, never()).requestRawJson(any(byte[].class), any(URI.class),
                eq(WhaleConstraintTreePublisher.UPDATE_PATH), eq(WorkerUpdateResponse.class));
    }

    private SerializedArtifact artifact() {
        byte[] payload = "{\"version\":7}".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        return new SerializedArtifact(
                new ArtifactMetadata(7, "gul_item", 1699, 151645, 2, 2, 4, 1, payload.length),
                payload);
    }
}
