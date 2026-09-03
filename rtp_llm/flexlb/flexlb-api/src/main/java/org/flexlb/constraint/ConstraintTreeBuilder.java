package org.flexlb.constraint;

import org.flexlb.constraint.ConstraintTreeModels.Artifact;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;

import java.time.Clock;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class ConstraintTreeBuilder implements AutoCloseable {

    private final Clock clock;
    private final ForkJoinPool buildPool;

    public ConstraintTreeBuilder() {
        this(Clock.systemUTC(), configuredBuildThreads());
    }

    ConstraintTreeBuilder(Clock clock) {
        this(clock, 1);
    }

    ConstraintTreeBuilder(Clock clock, int buildThreads) {
        this.clock = clock;
        if (buildThreads <= 0) {
            throw new IllegalArgumentException("buildThreads must be greater than zero");
        }
        this.buildPool = new ForkJoinPool(buildThreads,
                pool -> {
                    var worker = ForkJoinPool.defaultForkJoinWorkerThreadFactory.newThread(pool);
                    worker.setName("constraint-tree-build-" + worker.getPoolIndex());
                    return worker;
                },
                null,
                false);
    }

    private static int configuredBuildThreads() {
        int available = Runtime.getRuntime().availableProcessors();
        int safeDefault = Math.max(1, Math.min(10, available - Math.min(4, Math.max(1, available / 4))));
        String configured = System.getenv("CONSTRAINT_TREE_BUILD_THREADS");
        if (configured == null || configured.isBlank()) {
            return safeDefault;
        }
        try {
            return Math.max(1, Integer.parseInt(configured));
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("CONSTRAINT_TREE_BUILD_THREADS must be an integer", e);
        }
    }

    public void validateMetadata(BuildRequest request) {
        if (request == null) {
            throw new IllegalArgumentException("request must not be null");
        }
        if (request.version() <= 0) {
            throw new IllegalArgumentException("version must be greater than zero");
        }
        if (request.resolvedStartTokenId() < 0 || request.resolvedEndTokenId() < 0) {
            throw new IllegalArgumentException("start_token_id and end_token_id must not be negative");
        }
        if (request.resolvedStartTokenId() == request.resolvedEndTokenId()) {
            throw new IllegalArgumentException("start_token_id and end_token_id must be different");
        }
        if (request.resolvedSeparator().isEmpty()) {
            throw new IllegalArgumentException("sep must not be empty");
        }
        if (request.model() == null || request.model().isBlank()) {
            throw new IllegalArgumentException("model must not be blank");
        }
        if (request.hasRqTokenIds() == request.hasSids()) {
            throw new IllegalArgumentException("exactly one of rq_token_ids or sids must be non-empty");
        }
    }

    public Artifact build(BuildRequest request) {
        validateMetadata(request);

        String separator = request.resolvedSeparator();
        int startTokenId = request.resolvedStartTokenId();
        int endTokenId = request.resolvedEndTokenId();
        int[][] tokenSequences = parseAndSort(request, separator, startTokenId, endTokenId);
        Map<String, List<Integer>> prefixDict = new LinkedHashMap<>();
        long uniqueSidCount = 0;
        int[] previous = null;

        // Lexicographic sorting makes all candidates for the same prefix contiguous.
        // We can therefore append unique edges directly, without a HashSet per prefix.
        for (int[] tokenIds : tokenSequences) {
            if (previous != null && Arrays.equals(previous, tokenIds)) {
                continue;
            }
            uniqueSidCount++;
            StringBuilder prefix = new StringBuilder(Integer.toString(startTokenId));
            for (int tokenId : tokenIds) {
                appendCandidate(prefixDict, prefix.toString(), tokenId);
                prefix.append(separator).append(tokenId);
            }
            appendCandidate(prefixDict, prefix.toString(), endTokenId);
            previous = tokenIds;
        }

        prefixDict.replaceAll((ignored, candidates) -> List.copyOf(candidates));

        return new Artifact(
                request.version(),
                request.model(),
                startTokenId,
                endTokenId,
                separator,
                Collections.unmodifiableMap(prefixDict),
                request.inputCount(),
                uniqueSidCount,
                prefixDict.size(),
                clock.millis());
    }

    private int[][] parseAndSort(BuildRequest request, String separator, int startTokenId, int endTokenId) {
        int[][] tokenSequences = new int[request.inputCount()][];
        runOnBuildPool(() -> IntStream.range(0, tokenSequences.length).parallel().forEach(index -> {
            int[] tokens = request.hasRqTokenIds()
                    ? copyAndValidateTokens(request.rqTokenIds().get(index), index, startTokenId, endTokenId)
                    : parseSid(request.sids().get(index), separator, index, startTokenId, endTokenId);
            tokenSequences[index] = tokens;
        }));
        runOnBuildPool(() -> Arrays.parallelSort(tokenSequences, ConstraintTreeBuilder::compareTokenSequences));
        return tokenSequences;
    }

    private static int compareTokenSequences(int[] left, int[] right) {
        int commonLength = Math.min(left.length, right.length);
        for (int index = 0; index < commonLength; index++) {
            int comparison = Integer.compare(left[index], right[index]);
            if (comparison != 0) {
                return comparison;
            }
        }
        return Integer.compare(left.length, right.length);
    }

    private void runOnBuildPool(Runnable task) {
        try {
            buildPool.submit(task).get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("constraint tree build was interrupted", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            while (cause.getCause() != null) {
                cause = cause.getCause();
            }
            if (cause instanceof RuntimeException runtimeException) {
                throw runtimeException;
            }
            throw new IllegalStateException("constraint tree build failed", cause);
        }
    }

    private void appendCandidate(Map<String, List<Integer>> prefixDict, String prefix, int tokenId) {
        List<Integer> candidates = prefixDict.computeIfAbsent(prefix, ignored -> new ArrayList<>());
        if (candidates.isEmpty() || candidates.get(candidates.size() - 1) != tokenId) {
            candidates.add(tokenId);
        }
    }

    private int[] copyAndValidateTokens(int[] tokenIds,
                                        int sidIndex,
                                        int startTokenId,
                                        int endTokenId) {
        if (tokenIds == null || tokenIds.length == 0) {
            throw invalidSid(sidIndex, "SID token sequence must not be empty");
        }
        int[] copy = Arrays.copyOf(tokenIds, tokenIds.length);
        for (int tokenId : copy) {
            validateTokenId(tokenId, sidIndex, startTokenId, endTokenId);
        }
        return copy;
    }

    private int[] parseSid(String sid,
                           String separator,
                           int sidIndex,
                           int startTokenId,
                           int endTokenId) {
        if (sid == null || sid.isEmpty()) {
            throw invalidSid(sidIndex, "SID must not be empty");
        }

        List<Integer> tokenIds = new ArrayList<>();
        int tokenStart = 0;
        while (tokenStart <= sid.length()) {
            int separatorIndex = sid.indexOf(separator, tokenStart);
            int tokenEnd = separatorIndex < 0 ? sid.length() : separatorIndex;
            if (tokenEnd == tokenStart) {
                throw invalidSid(sidIndex, "SID contains an empty token id");
            }
            int tokenId;
            try {
                tokenId = Integer.parseInt(sid, tokenStart, tokenEnd, 10);
            } catch (NumberFormatException e) {
                throw invalidSid(sidIndex, "SID contains a non-int32 token id");
            }
            validateTokenId(tokenId, sidIndex, startTokenId, endTokenId);
            tokenIds.add(tokenId);
            if (separatorIndex < 0) {
                break;
            }
            tokenStart = tokenEnd + separator.length();
        }
        return tokenIds.stream().mapToInt(Integer::intValue).toArray();
    }

    private void validateTokenId(int tokenId, int sidIndex, int startTokenId, int endTokenId) {
        if (tokenId < 0) {
            throw invalidSid(sidIndex, "SID token ids must not be negative");
        }
        if (tokenId == startTokenId || tokenId == endTokenId) {
            throw invalidSid(sidIndex, "SID must not contain reserved start/end token ids");
        }
    }

    private IllegalArgumentException invalidSid(int sidIndex, String reason) {
        return new IllegalArgumentException("invalid SID at index " + sidIndex + ": " + reason);
    }

    @Override
    public void close() {
        buildPool.shutdownNow();
    }
}
