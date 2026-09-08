package org.flexlb.constraint;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.constraint.ConstraintTreeModels.Artifact;
import org.flexlb.constraint.ConstraintTreeModels.ArtifactMetadata;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;
import org.flexlb.constraint.ConstraintTreeModels.BuildState;
import org.flexlb.constraint.ConstraintTreeModels.BuildStatus;
import org.flexlb.constraint.ConstraintTreeModels.PublicationResult;
import org.flexlb.constraint.ConstraintTreeModels.SerializedArtifact;
import org.flexlb.constraint.ConstraintTreeModels.Submission;
import org.flexlb.constraint.ConstraintTreeModels.SubmissionState;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.PreDestroy;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

@Service
@Slf4j
public class ConstraintTreeBuildService {

    private final ConstraintTreeBuilder builder;
    private final ExecutorService buildExecutor;
    private final ConstraintTreePublisher publisher;
    private final ScheduledExecutorService reconcileExecutor;
    private final AtomicLong latestAcceptedVersion = new AtomicLong();
    private volatile String latestContentSha256 = "";
    private volatile String latestModel = "";
    private BuildRequest retryRequest;
    private final AtomicReference<SerializedArtifact> currentArtifact = new AtomicReference<>();
    private final AtomicReference<SerializedArtifact> backupArtifact = new AtomicReference<>();
    private final AtomicReference<BuildStatus> status = new AtomicReference<>(
            new BuildStatus(BuildState.IDLE, 0, 0, 0, 0, 0, 0, 0,
                    "no build has been submitted"));

    public ConstraintTreeBuildService() {
        this(new ConstraintTreeBuilder(), createBuildExecutor(),
                artifact -> new PublicationResult(0, 0, List.of()), null);
    }

    @Autowired
    public ConstraintTreeBuildService(WhaleConstraintTreePublisher publisher) {
        this(new ConstraintTreeBuilder(), createBuildExecutor(), publisher,
                Executors.newSingleThreadScheduledExecutor(new NamedThreadFactory("constraint-tree-reconciler")));
        long intervalSeconds = configuredReconcileIntervalSeconds();
        reconcileExecutor.scheduleWithFixedDelay(this::reconcileCurrent,
                intervalSeconds, intervalSeconds, TimeUnit.SECONDS);
    }

    ConstraintTreeBuildService(ConstraintTreeBuilder builder, ExecutorService buildExecutor) {
        this(builder, buildExecutor, artifact -> new PublicationResult(0, 0, List.of()), null);
    }

    ConstraintTreeBuildService(ConstraintTreeBuilder builder,
                               ExecutorService buildExecutor,
                               ConstraintTreePublisher publisher) {
        this(builder, buildExecutor, publisher, null);
    }

    ConstraintTreeBuildService(ConstraintTreeBuilder builder,
                               ExecutorService buildExecutor,
                               ConstraintTreePublisher publisher,
                               ScheduledExecutorService reconcileExecutor) {
        this.builder = builder;
        this.buildExecutor = buildExecutor;
        this.publisher = publisher;
        this.reconcileExecutor = reconcileExecutor;
    }

    private static ExecutorService createBuildExecutor() {
        return new ThreadPoolExecutor(
                1,
                1,
                0,
                TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(1),
                new NamedThreadFactory("constraint-tree-builder"),
                new ThreadPoolExecutor.DiscardOldestPolicy());
    }

    private static long configuredReconcileIntervalSeconds() {
        String value = System.getenv("CONSTRAINT_TREE_RECONCILE_INTERVAL_SECONDS");
        if (value == null || value.isBlank()) {
            return 60;
        }
        try {
            return Math.max(1, Long.parseLong(value));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("CONSTRAINT_TREE_RECONCILE_INTERVAL_SECONDS must be an integer", e);
        }
    }

    public Submission submit(BuildRequest request) {
        builder.validateMetadata(request);
        BuildRequest frozen = ConstraintTreeRequestIdentity.freeze(request);
        String contentSha256 = ConstraintTreeRequestIdentity.fingerprint(frozen);
        return submitFrozen(frozen, contentSha256, false);
    }

    private synchronized Submission submitFrozen(BuildRequest request, String contentSha256, boolean manualRetry) {
        long latestVersion = latestAcceptedVersion.get();
        if (request.version() < latestVersion) {
            return new Submission(
                    SubmissionState.STALE_VERSION,
                    request.version(),
                    latestVersion,
                    "a newer version has already been accepted");
        }
        if (request.version() == latestVersion && !contentSha256.equals(latestContentSha256)) {
            return new Submission(SubmissionState.VERSION_CONFLICT, request.version(), latestVersion,
                    "same version has different content; submit a new version");
        }
        if (request.version() == latestVersion && !manualRetry) {
            return new Submission(
                    SubmissionState.ALREADY_ACCEPTED,
                    request.version(),
                    latestVersion,
                    "this version has already been accepted");
        }

        latestAcceptedVersion.set(request.version());
        latestContentSha256 = contentSha256;
        latestModel = request.model();
        retryRequest = request;
        SerializedArtifact current = currentArtifact.get();
        SerializedArtifact backup = backupArtifact.get();
        setStatus(new BuildStatus(
                BuildState.QUEUED,
                request.version(),
                versionOf(current),
                versionOf(backup),
                request.inputCount(),
                current == null ? 0 : current.metadata().prefixCount(),
                0,
                0,
                "build queued"));
        buildExecutor.execute(() -> buildLatest(request, contentSha256));
        return new Submission(
                SubmissionState.ACCEPTED,
                request.version(),
                request.version(),
                "build accepted");
    }

    public void validateRequest(BuildRequest request) {
        builder.validateMetadata(request);
    }

    public synchronized Submission retry(ConstraintTreeModels.RetryRequest request) {
        if (request == null || request.version() != latestAcceptedVersion.get() || !latestModel.equals(request.model())) {
            throw new IllegalArgumentException("retry must identify the latest submitted model and version");
        }
        BuildState state = status.get().state();
        if (state == BuildState.BUILDING || state == BuildState.QUEUED || state == BuildState.PUBLISHING) {
            return new Submission(SubmissionState.ALREADY_ACCEPTED, request.version(), request.version(), "task is still running");
        }
        if (state == BuildState.FAILED && retryRequest != null) {
            return submitFrozen(retryRequest, latestContentSha256, true);
        }
        buildExecutor.execute(this::reconcileCurrent);
        return new Submission(SubmissionState.ACCEPTED, request.version(), request.version(),
                "republication queued; the existing artifact will not be rebuilt");
    }

    private void buildLatest(BuildRequest request, String contentSha256) {
        if (request.version() != latestAcceptedVersion.get()) {
            return;
        }
        SerializedArtifact current = currentArtifact.get();
        SerializedArtifact backup = backupArtifact.get();
        setStatus(new BuildStatus(
                BuildState.BUILDING,
                request.version(),
                versionOf(current),
                versionOf(backup),
                request.inputCount(),
                current == null ? 0 : current.metadata().prefixCount(),
                0,
                0,
                "building prefix tree"));
        long buildStartedAt = System.nanoTime();
        try {
            ConstraintTreeModels.PreparedBuild prepared = publisher.prepare(request);
            Artifact built = builder.build(prepared.request());
            if (request.version() != latestAcceptedVersion.get()) {
                return;
            }
            byte[] payload = ConstraintTreeCsrCodec.encode(built, prepared.mappingFingerprint(), contentSha256);
            SerializedArtifact candidate = new SerializedArtifact(
                    new ArtifactMetadata(
                            built.version(),
                            built.model(),
                            built.startTokenId(),
                            built.endTokenId(),
                            built.inputSidCount(),
                            built.sidCount(),
                            built.prefixCount(),
                            built.edgeCount(),
                            built.createdAtEpochMs(),
                            payload.length),
                    payload, prepared.mappingFingerprint(), contentSha256);

            SerializedArtifact previous;
            synchronized (this) {
                if (request.version() != latestAcceptedVersion.get()) {
                    return;
                }
                previous = currentArtifact.getAndSet(candidate);
                backupArtifact.set(previous);
                retryRequest = null; // Re-push the artifact after publication errors; do not retain all input SIDs.
            }
            long buildMillis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - buildStartedAt);
            log.info("constraint CSR built model={}, version={}, input_sids={}, unique_sids={}, states={}, edges={}, bytes={}, cost_ms={}",
                    built.model(), built.version(), built.inputSidCount(), built.sidCount(), built.prefixCount(),
                    built.edgeCount(), payload.length, buildMillis);

            setStatus(new BuildStatus(
                    BuildState.PUBLISHING,
                    request.version(),
                    candidate.version(),
                    versionOf(previous),
                    built.sidCount(),
                    built.prefixCount(),
                    0,
                    0,
                    "publishing tree to Whale inference workers"));

            PublicationResult publication;
            try {
                publication = publisher.publish(candidate);
            } catch (Exception publishError) {
                if (request.version() != latestAcceptedVersion.get()) {
                    return;
                }
                log.warn("constraint tree built but publication failed model={}, version={}",
                        request.model(), request.version(), publishError);
                setStatus(new BuildStatus(
                        BuildState.PARTIALLY_PUBLISHED,
                        request.version(),
                        candidate.version(),
                        versionOf(previous),
                        built.sidCount(),
                        built.prefixCount(),
                        0,
                        0,
                        "tree built; publication failed and will be retried: " + rootMessage(publishError)));
                return;
            }
            if (request.version() != latestAcceptedVersion.get()) {
                return;
            }
            boolean fullyPublished = publication.fullyPublished();
            String publicationMessage;
            if (publication.targetWorkerCount() == 0) {
                publicationMessage = "tree built; no Whale inference workers discovered yet";
            } else if (fullyPublished) {
                publicationMessage = "tree built and activated by all discovered workers";
            } else {
                publicationMessage = "tree built; some Whale inference workers still need retry";
            }
            setStatus(new BuildStatus(
                    fullyPublished ? BuildState.READY : BuildState.PARTIALLY_PUBLISHED,
                    request.version(),
                    candidate.version(),
                    versionOf(previous),
                    built.sidCount(),
                    built.prefixCount(),
                    publication.publishedWorkerCount(),
                    publication.targetWorkerCount(),
                    publicationMessage));
        } catch (Exception e) {
            if (request.version() != latestAcceptedVersion.get()) {
                return;
            }
            SerializedArtifact active = currentArtifact.get();
            SerializedArtifact previous = backupArtifact.get();
            log.error("constraint tree build failed model={}, version={}", request.model(), request.version(), e);
            setStatus(new BuildStatus(
                    BuildState.FAILED,
                    request.version(),
                    versionOf(active),
                    versionOf(previous),
                    request.inputCount(),
                    active == null ? 0 : active.metadata().prefixCount(),
                    0,
                    0,
                    e.getMessage()));
        }
    }

    private static long versionOf(SerializedArtifact artifact) {
        return artifact == null ? 0 : artifact.version();
    }

    void reconcileCurrent() {
        try {
            BuildStatus before = status.get();
            if (before.state() == BuildState.BUILDING
                    || before.state() == BuildState.QUEUED
                    || before.state() == BuildState.PUBLISHING) {
                return;
            }
            SerializedArtifact current = currentArtifact.get();
            if (current == null) {
                return;
            }
            long latestVersionBeforePublish = latestAcceptedVersion.get();
            PublicationResult publication = publisher.publish(current);
            if (currentArtifact.get() != current
                    || latestAcceptedVersion.get() != latestVersionBeforePublish) {
                return;
            }
            // A failed newer build must not prevent a restarted Worker from
            // receiving the last known-good snapshot. Keep the FAILED status,
            // however, so that submitting the failed version again remains legal.
            if (latestVersionBeforePublish != current.version()) {
                log.info("republished last known-good constraint tree after newer build failure "
                                + "current_version={}, latest_requested_version={}, workers={}/{}",
                        current.version(), latestVersionBeforePublish,
                        publication.publishedWorkerCount(), publication.targetWorkerCount());
                return;
            }
            boolean complete = publication.fullyPublished();
            String publicationMessage;
            if (publication.targetWorkerCount() == 0) {
                publicationMessage = "no Whale inference workers discovered yet";
            } else if (complete) {
                publicationMessage = "all discovered workers activated the current tree";
            } else {
                publicationMessage = "some Whale inference workers still need retry";
            }
            setStatus(new BuildStatus(
                    complete ? BuildState.READY : BuildState.PARTIALLY_PUBLISHED,
                    current.version(),
                    current.version(),
                    versionOf(backupArtifact.get()),
                    current.metadata().sidCount(),
                    current.metadata().prefixCount(),
                    publication.publishedWorkerCount(),
                    publication.targetWorkerCount(),
                    publicationMessage));
        } catch (Exception e) {
            log.warn("constraint tree reconciliation failed", e);
        }
    }

    private synchronized void setStatus(BuildStatus next) {
        if (next.requestedVersion() == latestAcceptedVersion.get()) {
            status.set(next);
        }
    }

    public synchronized BuildStatus getStatus() {
        BuildStatus current = status.get();
        SerializedArtifact artifact = currentArtifact.get();
        return new BuildStatus(current.state(), current.requestedVersion(), current.activeVersion(), current.backupVersion(),
                current.sidCount(), current.prefixCount(), current.publishedWorkerCount(), current.targetWorkerCount(),
                current.message(), latestModel, artifact == null ? "" : artifact.mappingFingerprint(),
                latestContentSha256, artifact == null ? "" : artifact.contentSha256());
    }

    public Optional<SerializedArtifact> getCurrentArtifact() {
        return Optional.ofNullable(currentArtifact.get());
    }

    public Optional<SerializedArtifact> getBackupArtifact() {
        return Optional.ofNullable(backupArtifact.get());
    }

    @PreDestroy
    public void destroy() {
        buildExecutor.shutdownNow();
        if (reconcileExecutor != null) {
            reconcileExecutor.shutdownNow();
        }
        builder.close();
    }

    private static String rootMessage(Throwable throwable) {
        Throwable current = throwable;
        while (current.getCause() != null) {
            current = current.getCause();
        }
        return current.getMessage() == null ? current.getClass().getSimpleName() : current.getMessage();
    }

}
