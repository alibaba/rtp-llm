package org.flexlb.constraint;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

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
            @JsonProperty("row_ptr") int[] rowPtr,
            @JsonProperty("col_idx") int[] colIdx,
            @JsonProperty("next_state") int[] nextState,
            @JsonProperty("input_sid_count") long inputSidCount,
            @JsonProperty("sid_count") long sidCount,
            @JsonProperty("created_at_epoch_ms") long createdAtEpochMs) {

        public long prefixCount() {
            return rowPtr.length - 1L;
        }

        public long edgeCount() {
            return colIdx.length;
        }
    }

    public record ArtifactMetadata(
            long version,
            String model,
            @JsonProperty("start_token_id") int startTokenId,
            @JsonProperty("end_token_id") int endTokenId,
            @JsonProperty("input_sid_count") long inputSidCount,
            @JsonProperty("sid_count") long sidCount,
            @JsonProperty("prefix_count") long prefixCount,
            @JsonProperty("edge_count") long edgeCount,
            @JsonProperty("created_at_epoch_ms") long createdAtEpochMs,
            @JsonProperty("serialized_size_bytes") long serializedSizeBytes) {
    }

    public record SerializedArtifact(ArtifactMetadata metadata, byte[] payload,
                                     String mappingFingerprint, String contentSha256) {
        public SerializedArtifact(ArtifactMetadata metadata, byte[] payload) {
            this(metadata, payload, "", "");
        }
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
            String message,
            String model,
            @JsonProperty("mapping_fingerprint") String mappingFingerprint,
            @JsonProperty("content_sha256") String contentSha256,
            @JsonProperty("active_content_sha256") String activeContentSha256) {
        public BuildStatus(BuildState state, long requestedVersion, long activeVersion, long backupVersion,
                           long sidCount, long prefixCount, int publishedWorkerCount, int targetWorkerCount, String message) {
            this(state, requestedVersion, activeVersion, backupVersion, sidCount, prefixCount,
                    publishedWorkerCount, targetWorkerCount, message, "", "", "", "");
        }
    }

    public enum SubmissionState {
        ACCEPTED,
        ALREADY_ACCEPTED,
        STALE_VERSION,
        VERSION_CONFLICT
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
            @JsonProperty("prefix_count") long prefixCount,
            @JsonProperty("edge_count") long edgeCount,
            @JsonProperty("mapping_fingerprint") String mappingFingerprint,
            @JsonProperty("content_sha256") String contentSha256) {
        public WorkerUpdateResponse(String status, long version, long requestedVersion, String message,
                                    boolean initialized, long prefixCount, long edgeCount) {
            this(status, version, requestedVersion, message, initialized, prefixCount, edgeCount, "", "");
        }
    }

    public record RetryRequest(long version, String model) { }

    public record PreparedBuild(BuildRequest request, String mappingFingerprint) { }

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
