package org.flexlb.constraint;

import org.flexlb.constraint.ConstraintTreeModels.Artifact;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;

import java.time.Clock;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

public class ConstraintTreeBuilder implements AutoCloseable {

    static final int TERMINAL_STATE = -1;

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
        CsrArrays csr = buildCsr(tokenSequences, endTokenId);

        return new Artifact(
                request.version(),
                request.model(),
                startTokenId,
                endTokenId,
                csr.rowPtr(),
                csr.colIdx(),
                csr.nextState(),
                request.inputCount(),
                csr.uniqueSidCount(),
                clock.millis());
    }

    /**
     * Builds the trie directly in CSR form. The input is lexicographically sorted, so
     * state ids can be assigned deterministically with only a stack for the previous
     * path. No String prefix keys or per-node HashMaps are materialized.
     */
    private CsrArrays buildCsr(int[][] tokenSequences, int endTokenId) {
        long stateCountLong = 1; // state 0 is the root/start-token state
        long uniqueSidCount = 0;
        int maxDepth = 0;
        int[] previous = null;
        for (int[] tokens : tokenSequences) {
            if (previous != null && Arrays.equals(previous, tokens)) {
                continue;
            }
            int commonPrefix = commonPrefixLength(previous, tokens);
            stateCountLong += tokens.length - commonPrefix;
            uniqueSidCount++;
            maxDepth = Math.max(maxDepth, tokens.length);
            previous = tokens;
        }
        long edgeCountLong = stateCountLong - 1 + uniqueSidCount;
        if (stateCountLong > Integer.MAX_VALUE || edgeCountLong > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("constraint tree exceeds int32 CSR capacity");
        }

        int stateCount = (int) stateCountLong;
        int edgeCount = (int) edgeCountLong;
        int[] degrees = new int[stateCount];
        assignStates(tokenSequences, maxDepth,
                (parent, token, child) -> degrees[parent]++,
                terminal -> degrees[terminal]++);

        int[] rowPtr = new int[stateCount + 1];
        for (int state = 0; state < stateCount; state++) {
            rowPtr[state + 1] = Math.addExact(rowPtr[state], degrees[state]);
        }
        if (rowPtr[stateCount] != edgeCount) {
            throw new IllegalStateException("CSR edge count mismatch");
        }

        int[] colIdx = new int[edgeCount];
        int[] nextState = new int[edgeCount];
        int[] cursor = Arrays.copyOf(rowPtr, stateCount);
        assignStates(tokenSequences, maxDepth,
                (parent, token, child) -> {
                    int position = cursor[parent]++;
                    colIdx[position] = token;
                    nextState[position] = child;
                },
                terminal -> {
                    int position = cursor[terminal]++;
                    colIdx[position] = endTokenId;
                    nextState[position] = TERMINAL_STATE;
                });

        // Normal child edges arrive sorted because the SID list is sorted. A terminal
        // edge can share a row with longer SIDs, so sort each row together with its
        // next-state value to make Worker-side binary search valid for every token id.
        runOnBuildPool(() -> IntStream.range(0, stateCount).parallel().forEach(state ->
                sortEdgePairs(colIdx, nextState, rowPtr[state], rowPtr[state + 1])));
        return new CsrArrays(rowPtr, colIdx, nextState, uniqueSidCount);
    }

    private void assignStates(int[][] tokenSequences,
                              int maxDepth,
                              EdgeConsumer edgeConsumer,
                              TerminalConsumer terminalConsumer) {
        int[] stateStack = new int[maxDepth + 1];
        int nextStateId = 1;
        int[] previous = null;
        for (int[] tokens : tokenSequences) {
            if (previous != null && Arrays.equals(previous, tokens)) {
                continue;
            }
            int commonPrefix = commonPrefixLength(previous, tokens);
            for (int depth = commonPrefix; depth < tokens.length; depth++) {
                int childState = nextStateId++;
                edgeConsumer.accept(stateStack[depth], tokens[depth], childState);
                stateStack[depth + 1] = childState;
            }
            terminalConsumer.accept(stateStack[tokens.length]);
            previous = tokens;
        }
    }

    private static int commonPrefixLength(int[] left, int[] right) {
        if (left == null || right == null) {
            return 0;
        }
        int limit = Math.min(left.length, right.length);
        int index = 0;
        while (index < limit && left[index] == right[index]) {
            index++;
        }
        return index;
    }

    private static void sortEdgePairs(int[] tokens, int[] nextStates, int from, int to) {
        // Rows are normally tiny. Insertion sort avoids allocating one boxed Pair per
        // edge and keeps peak memory bounded for multi-million-state trees.
        for (int index = from + 1; index < to; index++) {
            int token = tokens[index];
            int nextState = nextStates[index];
            int cursor = index - 1;
            while (cursor >= from && tokens[cursor] > token) {
                tokens[cursor + 1] = tokens[cursor];
                nextStates[cursor + 1] = nextStates[cursor];
                cursor--;
            }
            tokens[cursor + 1] = token;
            nextStates[cursor + 1] = nextState;
        }
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

    private record CsrArrays(int[] rowPtr, int[] colIdx, int[] nextState, long uniqueSidCount) {
    }

    @FunctionalInterface
    private interface EdgeConsumer {
        void accept(int parent, int token, int child);
    }

    @FunctionalInterface
    private interface TerminalConsumer {
        void accept(int terminalState);
    }
}
