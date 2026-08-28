package org.flexlb.balance.scheduler;

import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;

/**
 * Projected-gradient-descent joint request-to-endpoint assignment optimizer,
 * translated from navi_sched's {@code CostSchedulerKernelInternal}. It minimizes
 * a latency/queue objective over the per-request assignment simplex and reads
 * out an integral node choice for every request.
 *
 * <h2>Memory layout</h2>
 * All weight/cost/gradient buffers use navi's <em>node-major</em> layout:
 * {@code index = nodeIdx * requestCount + reqIdx}. The projection workspace is
 * <em>request-major within a tile</em>: {@code workspace[offset * nodeCount + nodeIdx]},
 * so each request's node column is contiguous for the simplex threshold search.
 *
 * <h2>Determinism</h2>
 * Weight initialization is random (uniform in {@code [1, 2)} then normalized to
 * the simplex, exactly as navi). Construct with a fixed seed for reproducible
 * golden-file tests; the default constructor uses {@link ThreadLocalRandom}.
 *
 * <p>Not thread-safe: a single instance owns reusable scratch buffers and must
 * be confined to one scheduling thread (or pooled per thread).
 */
final class NaviPgdOptimizer {

    /** Projection tile width; mirrors navi {@code PROJECTION_REQUEST_TILE_SIZE}. */
    static final int TILE_SIZE = 16;

    /** Mirrors navi {@code CostSchedulerConfig::MIN_DYNAMIC_LOOP_COUNT}. */
    private static final int MIN_DYNAMIC_LOOP_COUNT = 2;

    // ---- objective / optimizer configuration (navi CostSchedulerConfig) ----
    private double lambda = 0.5;
    private double alpha = 512.0;
    private double alphaDecay = 1.0;
    private double minAlpha = 0.0;
    private int maxLoopCount = 10;
    private long timeBudgetUs = 0;

    /** Non-null enables deterministic initialization for golden-file tests. */
    private final Long fixedSeed;

    // ---- lazily grown scratch buffers ----
    private int capacityNodes = 0;
    private int capacityRequests = 0;

    private double[] weights;             // node-major [N*R]
    private double[] gradients;           // node-major [N*R]
    private double[] costs;               // node-major [N*R]
    private double[] workspace;           // request-major tile [TILE*N]
    private double[] inverseTokenCounts;  // [R]
    private int[] bestNodeIndexes;        // [R]
    private int[] candidateIndexes;       // [R]
    private double[] ngCost;              // [N]
    private double[] ngWeight;            // [N]
    private double[] ngInvToken;          // [N]

    // ---- fixed-size tile scratch (TILE_SIZE) ----
    private final double[] tileActiveSums = new double[TILE_SIZE];
    private final double[] tileMaxValues = new double[TILE_SIZE];
    private final int[] tileMaxIndexes = new int[TILE_SIZE];
    private final double[] tileThresholds = new double[TILE_SIZE];
    private final double[] tileProjectedSums = new double[TILE_SIZE];
    private final double[] thresholdHolder = new double[1];

    NaviPgdOptimizer() {
        this.fixedSeed = null;
    }

    NaviPgdOptimizer(long seed) {
        this.fixedSeed = seed;
    }

    void configure(double lambda, double alpha, double alphaDecay, double minAlpha,
                   int maxLoopCount, long timeBudgetUs) {
        this.lambda = lambda;
        this.alpha = alpha;
        this.alphaDecay = alphaDecay;
        this.minAlpha = minAlpha;
        this.maxLoopCount = maxLoopCount;
        this.timeBudgetUs = timeBudgetUs;
    }

    /** Result of one optimization: the chosen node per request plus diagnostics. */
    static final class OptimizeResult {
        private final int[] selectedNodeIndexes;
        private final double forwardValue;
        private final int loopCount;

        OptimizeResult(int[] selectedNodeIndexes, double forwardValue, int loopCount) {
            this.selectedNodeIndexes = selectedNodeIndexes;
            this.forwardValue = forwardValue;
            this.loopCount = loopCount;
        }

        int[] selectedNodeIndexes() {
            return selectedNodeIndexes;
        }

        double forwardValue() {
            return forwardValue;
        }

        int loopCount() {
            return loopCount;
        }
    }

    /**
     * Grow the reusable buffers so they hold at least {@code nodeCount} nodes and
     * {@code requestCount} requests. Buffers are only ever grown, never shrunk.
     */
    void ensureCapacity(int nodeCount, int requestCount) {
        if (nodeCount <= capacityNodes && requestCount <= capacityRequests) {
            return;
        }
        int newNodes = Math.max(nodeCount, capacityNodes);
        int newRequests = Math.max(requestCount, capacityRequests);
        int weightCapacity = newNodes * newRequests;
        weights = new double[weightCapacity];
        gradients = new double[weightCapacity];
        costs = new double[weightCapacity];
        workspace = new double[TILE_SIZE * newNodes];
        inverseTokenCounts = new double[newRequests];
        bestNodeIndexes = new int[newRequests];
        candidateIndexes = new int[newRequests];
        ngCost = new double[newNodes];
        ngWeight = new double[newNodes];
        ngInvToken = new double[newNodes];
        capacityNodes = newNodes;
        capacityRequests = newRequests;
    }

    /**
     * Fill the node-major cost matrix. Mirrors navi {@code fillCostsForNodeRange}:
     * for each node it takes the endpoint's latency parameters and, for each
     * request, computes the linear cost from the request's total tokens and the
     * per-(node, request) cache-hit tokens.
     *
     * @param cacheHitTokens node-major [N*R] cache-hit tokens
     */
    void fillCosts(int nodeCount, int requestCount, double[][] nodeLatencyParameters,
                   long[] requestTokenCounts, long[] cacheHitTokens) {
        for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
            double[] params = nodeLatencyParameters[nodeIndex];
            int rowOffset = nodeIndex * requestCount;
            for (int requestIndex = 0; requestIndex < requestCount; requestIndex++) {
                long cacheHit = cacheHitTokens[rowOffset + requestIndex];
                costs[rowOffset + requestIndex] =
                        NaviPrefillModel.calculateRequestLinearCost(
                                params, requestTokenCounts[requestIndex], cacheHit);
            }
        }
    }

    /**
     * Forward pass. Mirrors navi {@code forwardNodeRange}: outer loop over nodes,
     * inner loop over requests accumulating {@code requestCostSum}/{@code weightSum}/
     * {@code inverseTokenWeightSum}, then the non-linear latency and its two
     * objective terms. When {@code computeGradients} is set, the per-node gradient
     * coefficients ({@code ngCost}/{@code ngWeight}/{@code ngInvToken}) are stored
     * for the backward pass.
     *
     * @return the objective value {@code sum1 * firstScale + sum2 * secondScale}
     */
    double forward(int nodeCount, int requestCount, long totalTokenCount, double lambda,
                   double[][] nodeLatencyParameters, double[] nodeQueueWaitMs,
                   boolean computeGradients) {
        double sum1 = 0.0;
        double sum2 = 0.0;
        double firstScale = lambda / totalTokenCount;
        double secondScale = (1.0 - lambda) / requestCount;
        for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
            int rowOffset = nodeIndex * requestCount;
            double requestCostSum = 0.0;
            double weightSum = 0.0;
            double inverseTokenWeightSum = 0.0;
            for (int requestIndex = 0; requestIndex < requestCount; requestIndex++) {
                double weight = weights[rowOffset + requestIndex];
                requestCostSum += weight * costs[rowOffset + requestIndex];
                weightSum += weight;
                inverseTokenWeightSum += weight * inverseTokenCounts[requestIndex];
            }
            double[] latencyAndDerivative =
                    NaviPrefillModel.calculateLatencyAndDerivative(
                            requestCostSum, nodeLatencyParameters[nodeIndex]);
            double nodeCost = latencyAndDerivative[0];
            double nodeCostDerivative = latencyAndDerivative[1];
            double queueWait = nodeQueueWaitMs != null ? nodeQueueWaitMs[nodeIndex] : 0.0;
            double queueCost = nodeCost + queueWait;
            double firstTerm = nodeCost * weightSum;
            double secondTerm = queueCost * inverseTokenWeightSum;
            if (computeGradients) {
                ngCost[nodeIndex] = nodeCostDerivative
                        * (firstScale * weightSum + secondScale * inverseTokenWeightSum);
                ngWeight[nodeIndex] = firstScale * nodeCost;
                ngInvToken[nodeIndex] = secondScale * queueCost;
            }
            sum1 += firstTerm;
            sum2 += secondTerm;
        }
        return sum1 * firstScale + sum2 * secondScale;
    }

    /**
     * Backward pass. Mirrors navi {@code backwardNodeRange}:
     * {@code grad[n*R+r] = ngCost[n]*cost[n*R+r] + ngWeight[n]
     * + ngInvToken[n]*inverseTokenCounts[r]}.
     */
    void backward(int nodeCount, int requestCount) {
        for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
            int rowOffset = nodeIndex * requestCount;
            double gradCost = ngCost[nodeIndex];
            double gradWeight = ngWeight[nodeIndex];
            double gradInvToken = ngInvToken[nodeIndex];
            for (int requestIndex = 0; requestIndex < requestCount; requestIndex++) {
                gradients[rowOffset + requestIndex] =
                        gradCost * costs[rowOffset + requestIndex]
                                + gradWeight
                                + gradInvToken * inverseTokenCounts[requestIndex];
            }
        }
    }

    /**
     * Fused gradient step, simplex projection and argmax readout. Mirrors navi
     * {@code projectRequestRange} with an active gradient span. Per 16-request
     * tile: gather {@code w - alpha*g} into the workspace (tracking the column
     * sum and argmax), find the simplex threshold, scatter
     * {@code max(0, w - alpha*g - threshold)} back into the weights, then add the
     * residual mass to the argmax node and record it as the candidate choice.
     */
    boolean projectAndReadout(int nodeCount, int requestCount, double stepAlpha) {
        if (!Double.isFinite(stepAlpha) || stepAlpha <= 0.0) {
            return false;
        }
        for (int requestBegin = 0; requestBegin < requestCount; requestBegin += TILE_SIZE) {
            int tileSize = Math.min(TILE_SIZE, requestCount - requestBegin);
            for (int offset = 0; offset < tileSize; offset++) {
                tileActiveSums[offset] = 0.0;
                tileMaxValues[offset] = Double.NEGATIVE_INFINITY;
                tileMaxIndexes[offset] = 0;
            }
            // Gather the gradient-stepped weights into the request-major workspace.
            for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
                int rowOffset = nodeIndex * requestCount;
                for (int offset = 0; offset < tileSize; offset++) {
                    int requestIndex = requestBegin + offset;
                    double updated = weights[rowOffset + requestIndex]
                            - stepAlpha * gradients[rowOffset + requestIndex];
                    workspace[offset * nodeCount + nodeIndex] = updated;
                    tileActiveSums[offset] += updated;
                    if (updated > tileMaxValues[offset]) {
                        tileMaxValues[offset] = updated;
                        tileMaxIndexes[offset] = nodeIndex;
                    }
                }
            }
            for (int offset = 0; offset < tileSize; offset++) {
                if (!findSimplexThreshold(workspace, offset * nodeCount, nodeCount,
                        tileActiveSums[offset], thresholdHolder)) {
                    return false;
                }
                tileThresholds[offset] = thresholdHolder[0];
            }
            for (int offset = 0; offset < tileSize; offset++) {
                tileProjectedSums[offset] = 0.0;
            }
            // Scatter the projected (thresholded, clamped) weights back in place.
            for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
                int rowOffset = nodeIndex * requestCount;
                for (int offset = 0; offset < tileSize; offset++) {
                    int requestIndex = requestBegin + offset;
                    double projected = Math.max(0.0,
                            weights[rowOffset + requestIndex]
                                    - stepAlpha * gradients[rowOffset + requestIndex]
                                    - tileThresholds[offset]);
                    weights[rowOffset + requestIndex] = projected;
                    tileProjectedSums[offset] += projected;
                }
            }
            for (int offset = 0; offset < tileSize; offset++) {
                int requestIndex = requestBegin + offset;
                weights[tileMaxIndexes[offset] * requestCount + requestIndex] +=
                        1.0 - tileProjectedSums[offset];
                candidateIndexes[requestIndex] = tileMaxIndexes[offset];
            }
        }
        return true;
    }

    /**
     * Active-set simplex threshold search. Mirrors navi {@code findSimplexThreshold}
     * (L306-334): compacts the {@code > threshold} entries in place and iterates
     * until the active set is stable, then writes the threshold to {@code out[0]}.
     *
     * @return {@code true} when a stable threshold was found
     */
    static boolean findSimplexThreshold(double[] buffer, int bufferOffset, int count,
                                        double activeSum, double[] out) {
        if (buffer == null || count == 0 || !Double.isFinite(activeSum)) {
            return false;
        }
        int activeCount = count;
        while (activeCount > 0) {
            double threshold = (activeSum - 1.0) / activeCount;
            double nextSum = 0.0;
            int nextCount = 0;
            for (int index = 0; index < activeCount; index++) {
                double value = buffer[bufferOffset + index];
                if (value > threshold) {
                    buffer[bufferOffset + nextCount] = value;
                    nextSum += value;
                    nextCount++;
                }
            }
            if (nextCount == activeCount) {
                out[0] = threshold;
                return true;
            }
            activeSum = nextSum;
            activeCount = nextCount;
        }
        return false;
    }

    /** Mirrors navi {@code getStepAlpha}: decayed step floored at {@code minAlpha}. */
    static double getStepAlpha(double alpha, double alphaDecay, double minAlpha,
                               int iteration) {
        double stepAlpha = Math.max(minAlpha, alpha * Math.pow(alphaDecay, iteration));
        // navi uses std::numeric_limits<double>::min() (smallest positive normal).
        return stepAlpha > 0.0 ? stepAlpha : Double.MIN_NORMAL;
    }

    /** Mirrors navi {@code shouldRunNextLoop}: dynamic time-budget-aware stop. */
    static boolean shouldRunNextLoop(int completedLoopCount, int maxLoopCount,
                                     long optimizationTimeBudgetUs, long elapsedUs,
                                     long lastLoopUs) {
        if (completedLoopCount >= maxLoopCount) {
            return false;
        }
        int minimumLoopCount = Math.min(MIN_DYNAMIC_LOOP_COUNT, maxLoopCount);
        if (completedLoopCount < minimumLoopCount || optimizationTimeBudgetUs < 0) {
            return true;
        }
        long averageLoopUs = (elapsedUs + completedLoopCount - 1) / completedLoopCount;
        long nextLoopUs = Math.max(lastLoopUs, averageLoopUs);
        return elapsedUs < optimizationTimeBudgetUs
                && nextLoopUs <= optimizationTimeBudgetUs - elapsedUs;
    }

    /**
     * Run the full optimization. Mirrors navi {@code optimize}: seed the weights,
     * loop {forward -> record best -> backward -> project/readout -> decay alpha
     * -> time-budget check}, then a final gradient-free forward that may improve
     * on the best recorded assignment.
     *
     * @param nodeLatencyParameters per-node 9-parameter latency vectors
     * @param nodeQueueWaitMs       optional per-node queue wait (ms); may be null
     * @param requestTokenCounts    per-request prompt token counts (min 1 applied)
     * @param cacheHitTokens        node-major [N*R] cache-hit tokens
     * @return the assignment result, or {@code null} on a non-finite/invalid step
     */
    OptimizeResult optimize(int nodeCount, int requestCount,
                            double[][] nodeLatencyParameters, double[] nodeQueueWaitMs,
                            long[] requestTokenCounts, long[] cacheHitTokens) {
        if (nodeCount <= 0 || requestCount <= 0 || maxLoopCount <= 0) {
            return null;
        }
        ensureCapacity(nodeCount, requestCount);

        long totalTokenCount = 0;
        for (int requestIndex = 0; requestIndex < requestCount; requestIndex++) {
            long tokenCount = Math.max(1L, requestTokenCounts[requestIndex]);
            inverseTokenCounts[requestIndex] = 1.0 / tokenCount;
            totalTokenCount += tokenCount;
        }

        fillCosts(nodeCount, requestCount, nodeLatencyParameters,
                requestTokenCounts, cacheHitTokens);

        RandomGenerator random =
                fixedSeed != null ? new java.util.Random(fixedSeed) : ThreadLocalRandom.current();
        initializeWeights(requestCount, nodeCount, random);

        double bestLoss = Double.POSITIVE_INFINITY;
        int actualLoopCount = 0;
        long optimizationBeginUs = nowUs();
        long lastLoopEndUs = optimizationBeginUs;
        for (int iteration = 0; iteration < maxLoopCount; iteration++) {
            double stepAlpha = getStepAlpha(alpha, alphaDecay, minAlpha, iteration);
            double value = forward(nodeCount, requestCount, totalTokenCount, lambda,
                    nodeLatencyParameters, nodeQueueWaitMs, true);
            if (!Double.isFinite(value)) {
                return null;
            }
            if (iteration > 0 && value < bestLoss) {
                bestLoss = value;
                int[] swap = bestNodeIndexes;
                bestNodeIndexes = candidateIndexes;
                candidateIndexes = swap;
            }
            backward(nodeCount, requestCount);
            if (!projectAndReadout(nodeCount, requestCount, stepAlpha)) {
                return null;
            }
            actualLoopCount = iteration + 1;
            long stepEndUs = nowUs();
            long elapsedUs = stepEndUs - optimizationBeginUs;
            long lastLoopUs = stepEndUs - lastLoopEndUs;
            lastLoopEndUs = stepEndUs;
            if (!shouldRunNextLoop(actualLoopCount, maxLoopCount, timeBudgetUs,
                    elapsedUs, lastLoopUs)) {
                break;
            }
        }

        double finalForwardValue = forward(nodeCount, requestCount, totalTokenCount,
                lambda, nodeLatencyParameters, nodeQueueWaitMs, false);
        if (!Double.isFinite(finalForwardValue)) {
            return null;
        }
        if (finalForwardValue < bestLoss) {
            bestLoss = finalForwardValue;
            int[] swap = bestNodeIndexes;
            bestNodeIndexes = candidateIndexes;
            candidateIndexes = swap;
        }

        int[] selected = new int[requestCount];
        System.arraycopy(bestNodeIndexes, 0, selected, 0, requestCount);
        return new OptimizeResult(selected, bestLoss, actualLoopCount);
    }

    /**
     * Random simplex initialization. Mirrors navi {@code initializeWeights}: draw
     * uniform {@code [1, 2)} weights, accumulate per-request sums, and normalize
     * each request column onto the probability simplex via {@link #normalizeWeights}.
     */
    private void initializeWeights(int requestCount, int nodeCount, RandomGenerator random) {
        double[] randomWeightSums = new double[requestCount];
        for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
            boolean isLastNode = nodeIndex + 1 == nodeCount;
            int rowOffset = nodeIndex * requestCount;
            for (int requestIndex = 0; requestIndex < requestCount; requestIndex++) {
                double weight = 1.0 + random.nextDouble();  // uniform in [1, 2)
                weights[rowOffset + requestIndex] = weight;
                randomWeightSums[requestIndex] += weight;
                if (isLastNode) {
                    randomWeightSums[requestIndex] = 1.0 / randomWeightSums[requestIndex];
                }
            }
        }
        normalizeWeights(requestCount, nodeCount, randomWeightSums);
    }

    /**
     * Normalize each request column to sum to one. Mirrors navi
     * {@code normalizeWeights}: scale the first {@code nodeCount - 1} rows by the
     * inverse column sum, then set the last row to the residual mass so the column
     * lands exactly on the simplex.
     */
    private void normalizeWeights(int requestCount, int nodeCount, double[] inverseWeightSums) {
        double[] normalizedWeightSums = new double[requestCount];
        for (int nodeIndex = 0; nodeIndex + 1 < nodeCount; nodeIndex++) {
            int rowOffset = nodeIndex * requestCount;
            for (int requestIndex = 0; requestIndex < requestCount; requestIndex++) {
                double normalizedWeight =
                        weights[rowOffset + requestIndex] * inverseWeightSums[requestIndex];
                weights[rowOffset + requestIndex] = normalizedWeight;
                normalizedWeightSums[requestIndex] += normalizedWeight;
            }
        }
        int lastRowOffset = (nodeCount - 1) * requestCount;
        for (int requestIndex = 0; requestIndex < requestCount; requestIndex++) {
            weights[lastRowOffset + requestIndex] =
                    1.0 - normalizedWeightSums[requestIndex];
        }
    }

    private static long nowUs() {
        return System.nanoTime() / 1000L;
    }
}
