package org.flexlb.constraint;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

public final class ConstraintTreeModels {

    public static final int DEFAULT_START_TOKEN_ID = 1699;
    public static final int DEFAULT_END_TOKEN_ID = 151645;
    public static final String DEFAULT_SEPARATOR = "_";

    private ConstraintTreeModels() {
    }

    public record BuildRequest(
            long version,
            String model,
            @JsonProperty("start_token_id") Integer startTokenId,
            @JsonProperty("end_token_id") Integer endTokenId,
            @JsonProperty("sep") String separator,
            @JsonProperty("rq_token_ids") List<int[]> rqTokenIds,
            List<String> sids) {

        public BuildRequest(long version,
                            Integer startTokenId,
                            Integer endTokenId,
                            String separator,
                            List<String> sids) {
            this(version, null, startTokenId, endTokenId, separator, null, sids);
        }

        public int resolvedStartTokenId() {
            return startTokenId == null ? DEFAULT_START_TOKEN_ID : startTokenId;
        }

        public int resolvedEndTokenId() {
            return endTokenId == null ? DEFAULT_END_TOKEN_ID : endTokenId;
        }

        public String resolvedSeparator() {
            return separator == null ? DEFAULT_SEPARATOR : separator;
        }

        public boolean hasRqTokenIds() {
            return rqTokenIds != null && !rqTokenIds.isEmpty();
        }

        public boolean hasSids() {
            return sids != null && !sids.isEmpty();
        }

        public int inputCount() {
            if (hasRqTokenIds()) {
                return rqTokenIds.size();
            }
            return hasSids() ? sids.size() : 0;
        }
    }

    public record Artifact(
            long version,
            String model,
            @JsonProperty("start_token_id") int startTokenId,
            @JsonProperty("end_token_id") int endTokenId,
            @JsonProperty("sep") String separator,
            @JsonProperty("prefix_dict") Map<String, List<Integer>> prefixDict,
            @JsonProperty("input_sid_count") long inputSidCount,
            @JsonProperty("sid_count") long sidCount,
            @JsonProperty("prefix_count") long prefixCount,
            @JsonProperty("created_at_epoch_ms") long createdAtEpochMs) {
    }

    public record ArtifactMetadata(
            long version,
            String model,
            @JsonProperty("start_token_id") int startTokenId,
            @JsonProperty("end_token_id") int endTokenId,
            @JsonProperty("input_sid_count") long inputSidCount,
            @JsonProperty("sid_count") long sidCount,
            @JsonProperty("prefix_count") long prefixCount,
            @JsonProperty("created_at_epoch_ms") long createdAtEpochMs,
            @JsonProperty("serialized_size_bytes") long serializedSizeBytes) {
    }

    public record SerializedArtifact(ArtifactMetadata metadata, byte[] payload) {
        public long version() {
            return metadata.version();
        }
    }

    public enum BuildState {
        IDLE,
        QUEUED,
        BUILDING,
        PUBLISHING,
        READY,
        PARTIALLY_PUBLISHED,
        FAILED
    }

    public record BuildStatus(
            BuildState state,
            @JsonProperty("requested_version") long requestedVersion,
            @JsonProperty("active_version") long activeVersion,
            @JsonProperty("backup_version") long backupVersion,
            @JsonProperty("sid_count") long sidCount,
            @JsonProperty("prefix_count") long prefixCount,
            @JsonProperty("published_worker_count") int publishedWorkerCount,
            @JsonProperty("target_worker_count") int targetWorkerCount,
            String message) {
    }

    public enum SubmissionState {
        ACCEPTED,
        ALREADY_ACCEPTED,
        STALE_VERSION
    }

    public record Submission(
            SubmissionState state,
            @JsonProperty("requested_version") long requestedVersion,
            @JsonProperty("latest_version") long latestVersion,
            String message) {
    }

    public record WorkerUpdateResponse(
            String status,
            long version,
            @JsonProperty("requested_version") long requestedVersion,
            String message,
            boolean initialized,
            @JsonProperty("prefix_count") long prefixCount) {
    }

    public record WorkerPublication(
            String worker,
            boolean success,
            long version,
            String message) {
    }

    public record PublicationResult(
            @JsonProperty("target_worker_count") int targetWorkerCount,
            @JsonProperty("published_worker_count") int publishedWorkerCount,
            List<WorkerPublication> workers) {

        public boolean fullyPublished() {
            return targetWorkerCount > 0 && targetWorkerCount == publishedWorkerCount;
        }
    }
}
