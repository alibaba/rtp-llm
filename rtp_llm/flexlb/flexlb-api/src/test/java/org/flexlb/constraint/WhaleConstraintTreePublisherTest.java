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
import static org.mockito.Mockito.times;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
    void deliversToCppHttpPortsAndDefersSuccessUntilReconciliationObservesActivation() {
        WorkerHost first = new WorkerHost("10.0.0.1", 8000, 8001, 8005, "hz", "default");
        WorkerHost second = new WorkerHost("10.0.0.2", 9000, 9001, 9005, "sh", "default");
        when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of(first));
        when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of(first, second));
        when(http.get(any(URI.class), eq(WhaleConstraintTreePublisher.STATUS_PATH),
                eq(WorkerUpdateResponse.class))).thenReturn(Mono.empty());
        when(http.requestRawBytes(any(byte[].class), any(URI.class),
                eq(WhaleConstraintTreePublisher.UPDATE_PATH), eq(WorkerUpdateResponse.class)))
                .thenReturn(Mono.just(new WorkerUpdateResponse("accepted", 0, 7, "queued", false, 0, 0)));

        SerializedArtifact artifact = artifact();
        PublicationResult result = publisher.publish(artifact);

        assertEquals(2, result.targetWorkerCount());
        assertEquals(0, result.publishedWorkerCount());
        verify(http).requestRawBytes(eq(artifact.payload()), eq(URI.create("http://10.0.0.1:8005")),
                eq(WhaleConstraintTreePublisher.UPDATE_PATH), eq(WorkerUpdateResponse.class));
        verify(http).requestRawBytes(eq(artifact.payload()), eq(URI.create("http://10.0.0.2:9005")),
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
                .thenReturn(Mono.just(new WorkerUpdateResponse("ready", 7, 7, "ready", true, 4, 5)));

        PublicationResult result = publisher.publish(artifact());

        assertEquals(1, result.publishedWorkerCount());
        verify(http, never()).requestRawBytes(any(byte[].class), any(URI.class),
                eq(WhaleConstraintTreePublisher.UPDATE_PATH), eq(WorkerUpdateResponse.class));
    }

    @Test
    void newerWorkerVersionIsReportedAsConflictWithoutOverwritingIt() {
        WorkerHost worker = new WorkerHost("10.0.0.1", 8000, 8001, 8005, "hz", "default");
        when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of(worker));
        when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of());
        when(http.get(URI.create("http://10.0.0.1:8005"), WhaleConstraintTreePublisher.STATUS_PATH,
                WorkerUpdateResponse.class))
                .thenReturn(Mono.just(new WorkerUpdateResponse("ready", 8, 8, "ready", true, 4, 5)));

        PublicationResult result = publisher.publish(artifact());

        assertEquals(0, result.publishedWorkerCount());
        assertEquals(8, result.workers().get(0).version());
        verify(http, never()).requestRawBytes(any(byte[].class), any(URI.class),
                eq(WhaleConstraintTreePublisher.UPDATE_PATH), eq(WorkerUpdateResponse.class));
    }

    @Test
    void statusProbeDoesNotResendVersionThatWorkerIsAlreadyLoading() {
        WorkerHost worker = new WorkerHost("10.0.0.1", 8000, 8001, 8005, "hz", "default");
        when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of(worker));
        when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of());
        when(http.get(URI.create("http://10.0.0.1:8005"), WhaleConstraintTreePublisher.STATUS_PATH,
                WorkerUpdateResponse.class))
                .thenReturn(Mono.just(new WorkerUpdateResponse("loading", 6, 7, "loading", true, 3, 4)));

        PublicationResult result = publisher.publish(artifact());

        assertEquals(0, result.publishedWorkerCount());
        verify(http, never()).requestRawBytes(any(byte[].class), any(URI.class),
                eq(WhaleConstraintTreePublisher.UPDATE_PATH), eq(WorkerUpdateResponse.class));
    }

    private SerializedArtifact artifact() {
        byte[] payload = "{\"version\":7}".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        return new SerializedArtifact(
                new ArtifactMetadata(7, "gul_item", 1699, 151645, 2, 2, 4, 5, 1, payload.length),
                payload);
    }

    @Test
    void probesFingerprintsEveryBuildAndFetchesMappingOnlyWhenChanged() {
        WorkerHost worker = new WorkerHost("10.0.0.1", 8000, 8001, 8005, "hz", "default");
        URI uri = URI.create("http://10.0.0.1:8005");
        when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of(worker));
        when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of());
        var first = ConstraintTreeSidMappingTest.mapping(java.util.Map.of("C1", 17));
        var second = ConstraintTreeSidMappingTest.mapping(java.util.Map.of("C1", 19));
        when(http.get(uri, "/constraint_tree_mapping_status", ConstraintTreeSidMapping.class))
                .thenReturn(Mono.just(first), Mono.just(first), Mono.just(second));
        when(http.get(uri, "/constraint_tree_mapping", ConstraintTreeSidMapping.class))
                .thenReturn(Mono.just(first), Mono.just(second));
        assertEquals(17, publisher.prepare(ConstraintTreeSidMappingTest.request(1, "C1C1")).request().rqTokenIds().get(0)[0]);
        publisher.prepare(ConstraintTreeSidMappingTest.request(2, "C1C1"));
        assertEquals(19, publisher.prepare(ConstraintTreeSidMappingTest.request(3, "C1C1")).request().rqTokenIds().get(0)[0]);
        verify(http, times(3)).get(uri, "/constraint_tree_mapping_status", ConstraintTreeSidMapping.class);
        verify(http, times(2)).get(uri, "/constraint_tree_mapping", ConstraintTreeSidMapping.class);
    }

    @Test
    void mismatchedWorkersBlockConversionBeforeAnyFullMappingFetch() {
        WorkerHost first = new WorkerHost("10.0.0.1", 8000, 8001, 8005, "hz", "default");
        WorkerHost second = new WorkerHost("10.0.0.2", 9000, 9001, 9005, "sh", "default");
        when(addresses.getEngineWorkerList("gul_item", RoleType.DECODE)).thenReturn(List.of(first, second));
        when(addresses.getEngineWorkerList("gul_item", RoleType.PDFUSION)).thenReturn(List.of());
        when(http.get(any(URI.class), eq("/constraint_tree_mapping_status"), eq(ConstraintTreeSidMapping.class)))
                .thenReturn(Mono.just(ConstraintTreeSidMappingTest.mapping(java.util.Map.of("C1", 17))),
                        Mono.just(ConstraintTreeSidMappingTest.mapping(java.util.Map.of("C1", 19))));
        assertThrows(IllegalStateException.class, () -> publisher.prepare(ConstraintTreeSidMappingTest.request(1, "C1")));
        verify(http, never()).get(any(URI.class), eq("/constraint_tree_mapping"), eq(ConstraintTreeSidMapping.class));
    }
}
