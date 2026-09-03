package org.flexlb.constraint;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;
import org.flexlb.constraint.ConstraintTreeModels.BuildState;
import org.flexlb.constraint.ConstraintTreeModels.PublicationResult;
import org.flexlb.constraint.ConstraintTreeModels.Submission;
import org.flexlb.constraint.ConstraintTreeModels.SubmissionState;
import org.flexlb.constraint.ConstraintTreeModels.WorkerPublication;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class ConstraintTreeBuildServiceTest {

    private final ConstraintTreePublisher publisher = artifact -> new PublicationResult(
            1, 1, List.of(new WorkerPublication("127.0.0.1:8000", true, artifact.version(), "accepted")));
    private final ConstraintTreeBuildService service = new ConstraintTreeBuildService(
            new ConstraintTreeBuilder(), Executors.newSingleThreadExecutor(), publisher);

    @AfterEach
    void tearDown() {
        service.destroy();
    }

    @Test
    void buildsAsynchronouslyAndMakesSerializedArtifactAvailable() throws Exception {
        Submission submission = service.submit(request(10, "1_3", "1_4_5"));

        assertEquals(SubmissionState.ACCEPTED, submission.state());
        awaitState(BuildState.READY);
        assertEquals(10, service.getStatus().activeVersion());
        assertEquals(1, service.getStatus().publishedWorkerCount());

        JsonNode artifact = JsonUtils.toTreeNode(
                new String(service.getCurrentArtifact().orElseThrow().payload(), java.nio.charset.StandardCharsets.UTF_8));
        assertEquals(10, artifact.path("version").asLong());
        assertEquals(1699, artifact.path("start_token_id").asInt());
        assertEquals(151645, artifact.path("end_token_id").asInt());
        assertEquals(3, artifact.path("prefix_dict").path("1699_1").get(0).asInt());
        assertEquals(4, artifact.path("prefix_dict").path("1699_1").get(1).asInt());
    }

    @Test
    void retainsOnlyCurrentAndPreviousSuccessfulVersion() throws Exception {
        service.submit(request(10, "1_3"));
        awaitState(BuildState.READY);
        assertTrue(service.getBackupArtifact().isEmpty());

        service.submit(request(11, "4_5"));
        awaitState(BuildState.READY);

        assertEquals(11, service.getCurrentArtifact().orElseThrow().version());
        assertEquals(10, service.getBackupArtifact().orElseThrow().version());
        assertEquals(10, service.getStatus().backupVersion());
    }

    @Test
    void rejectsDuplicateAndStaleVersionsBeforeQueueing() throws Exception {
        service.submit(request(10, "1_3"));
        awaitState(BuildState.READY);

        assertEquals(SubmissionState.ALREADY_ACCEPTED, service.submit(request(10, "4_5_6")).state());
        assertEquals(SubmissionState.STALE_VERSION, service.submit(request(9, "4_5_6")).state());
        assertEquals(10, service.getStatus().activeVersion());
    }

    @Test
    void failedBuildKeepsCurrentAndBackupArtifacts() throws Exception {
        service.submit(request(10, "1_3"));
        awaitState(BuildState.READY);
        service.submit(request(11, "4_5"));
        awaitState(BuildState.READY);

        assertEquals(SubmissionState.ACCEPTED, service.submit(request(12, "malformed")).state());
        awaitState(BuildState.FAILED);

        assertEquals(12, service.getStatus().requestedVersion());
        assertEquals(11, service.getStatus().activeVersion());
        assertEquals(10, service.getStatus().backupVersion());
        assertTrue(service.getStatus().message().contains("invalid SID at index 0"));
        assertEquals(11, service.getCurrentArtifact().orElseThrow().version());
        assertEquals(10, service.getBackupArtifact().orElseThrow().version());

        assertEquals(SubmissionState.ACCEPTED, service.submit(request(12, "7_8_9")).state());
        awaitState(BuildState.READY);
        assertEquals(12, service.getCurrentArtifact().orElseThrow().version());
        assertEquals(11, service.getBackupArtifact().orElseThrow().version());
    }

    @Test
    void keepsBuiltArtifactAndMarksPartialWhenPublicationThrows() throws Exception {
        ConstraintTreeBuildService localService = new ConstraintTreeBuildService(
                new ConstraintTreeBuilder(),
                Executors.newSingleThreadExecutor(),
                artifact -> {
                    throw new IllegalStateException("service discovery unavailable");
                });
        try {
            localService.submit(request(20, "1_2_3"));
            awaitState(localService, BuildState.PARTIALLY_PUBLISHED);

            assertEquals(20, localService.getCurrentArtifact().orElseThrow().version());
            assertTrue(localService.getBackupArtifact().isEmpty());
            assertTrue(localService.getStatus().message().contains("service discovery unavailable"));
        } finally {
            localService.destroy();
        }
    }

    @Test
    void reportsPartialWhenNoWorkersAreDiscovered() throws Exception {
        ConstraintTreeBuildService localService = new ConstraintTreeBuildService(
                new ConstraintTreeBuilder(),
                Executors.newSingleThreadExecutor(),
                artifact -> new PublicationResult(0, 0, List.of()));
        try {
            localService.submit(request(21, "1_2_3"));
            awaitState(localService, BuildState.PARTIALLY_PUBLISHED);

            assertEquals(0, localService.getStatus().targetWorkerCount());
            assertTrue(localService.getStatus().message().contains("no Whale inference workers"));
        } finally {
            localService.destroy();
        }
    }

    @Test
    void rapidSubmissionsPublishOnlyTheLatestVersion() throws Exception {
        CountDownLatch firstBuildStarted = new CountDownLatch(1);
        CountDownLatch releaseFirstBuild = new CountDownLatch(1);
        ConstraintTreeBuilder blockingBuilder = new ConstraintTreeBuilder() {
            @Override
            public ConstraintTreeModels.Artifact build(BuildRequest request) {
                if (request.version() == 30) {
                    firstBuildStarted.countDown();
                    try {
                        if (!releaseFirstBuild.await(5, TimeUnit.SECONDS)) {
                            throw new IllegalStateException("timed out waiting to release first build");
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw new IllegalStateException("first build interrupted", e);
                    }
                }
                return super.build(request);
            }
        };
        List<Long> publishedVersions = new CopyOnWriteArrayList<>();
        ConstraintTreeBuildService localService = new ConstraintTreeBuildService(
                blockingBuilder,
                Executors.newSingleThreadExecutor(),
                artifact -> {
                    publishedVersions.add(artifact.version());
                    return new PublicationResult(1, 1, List.of());
                });
        try {
            localService.submit(request(30, "1_2"));
            assertTrue(firstBuildStarted.await(5, TimeUnit.SECONDS));
            localService.submit(request(31, "3_4"));
            localService.submit(request(32, "5_6_7"));
            releaseFirstBuild.countDown();
            awaitState(localService, BuildState.READY);

            assertEquals(32, localService.getCurrentArtifact().orElseThrow().version());
            assertTrue(localService.getBackupArtifact().isEmpty());
            assertEquals(List.of(32L), publishedVersions);
        } finally {
            releaseFirstBuild.countDown();
            localService.destroy();
        }
    }

    @Test
    void reconciliationRetriesTheCurrentArtifactUntilWorkersAcceptIt() throws Exception {
        AtomicInteger attempts = new AtomicInteger();
        ConstraintTreeBuildService localService = new ConstraintTreeBuildService(
                new ConstraintTreeBuilder(),
                Executors.newSingleThreadExecutor(),
                artifact -> attempts.incrementAndGet() == 1
                        ? new PublicationResult(1, 0, List.of())
                        : new PublicationResult(1, 1, List.of()));
        try {
            localService.submit(request(40, "1_2_3"));
            awaitState(localService, BuildState.PARTIALLY_PUBLISHED);

            localService.reconcileCurrent();

            assertEquals(BuildState.READY, localService.getStatus().state());
            assertEquals(2, attempts.get());
            assertEquals(40, localService.getStatus().activeVersion());
        } finally {
            localService.destroy();
        }
    }

    private BuildRequest request(long version, String... sids) {
        return new BuildRequest(version, "gul_item", null, null, null, null, List.of(sids));
    }

    private void awaitState(BuildState expected) throws Exception {
        awaitState(service, expected);
    }

    private void awaitState(ConstraintTreeBuildService target, BuildState expected) throws Exception {
        long deadline = System.nanoTime() + Duration.ofSeconds(5).toNanos();
        while (System.nanoTime() < deadline) {
            if (target.getStatus().state() == expected) {
                return;
            }
            Thread.sleep(5);
        }
        fail("timed out waiting for state " + expected + ", current status=" + target.getStatus());
    }
}
