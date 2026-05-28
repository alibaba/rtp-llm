package org.flexlb.balance.dp;

import io.grpc.ManagedChannel;
import io.grpc.netty.NettyChannelBuilder;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
public class PrefillProfiler {

    private final ConfigService configService;
    private final EngineWorkerStatus engineWorkerStatus;

    public PrefillProfiler(ConfigService configService, EngineWorkerStatus engineWorkerStatus) {
        this.configService = configService;
        this.engineWorkerStatus = engineWorkerStatus;
    }

    public void runIfEnabled() {
        FlexlbConfig config = configService.loadBalanceConfig();

        String manualCoeffs = config.getPrefillCoefficients();
        if (manualCoeffs != null && !manualCoeffs.isBlank()) {
            applyManualCoefficients(manualCoeffs);
            return;
        }

        if (!config.isPrefillProfilingEnabled()) {
            log.info("Prefill profiling disabled");
            return;
        }

        try {
            profile(config);
        } catch (Exception e) {
            log.warn("Prefill profiling failed, keeping default coefficients", e);
        }
    }

    private void applyManualCoefficients(String csv) {
        String[] parts = csv.split(",");
        if (parts.length < 3) {
            log.warn("prefillCoefficients must be 'c0,c1,c2', got '{}'; ignoring", csv);
            return;
        }
        double c0 = Double.parseDouble(parts[0].trim());
        double c1 = Double.parseDouble(parts[1].trim());
        double c2 = Double.parseDouble(parts[2].trim());
        TaskInfo.updateCoefficients(c0, c1, c2);
        log.info("Prefill coefficients set from config: c0={}, c1={}, c2={}", c0, c1, c2);
    }

    private void profile(FlexlbConfig config) throws InterruptedException {
        WorkerStatus worker = waitForPrefillWorker(config.getPrefillProfilingTimeoutMs());
        if (worker == null) {
            log.warn("No PREFILL worker found within {}ms, skipping profiling",
                    config.getPrefillProfilingTimeoutMs());
            return;
        }

        int[] tokenLengths = parseTokenLengths(config.getPrefillProfilingTokenLengths());
        int repeats = Math.max(1, config.getPrefillProfilingRepeats());

        int grpcPort = CommonUtils.toGrpcPort(worker.getPort());
        log.info("Prefill profiling: worker={}:{}, lengths={}, repeats={}",
                worker.getIp(), grpcPort, Arrays.toString(tokenLengths), repeats);

        ManagedChannel channel = NettyChannelBuilder.forAddress(worker.getIp(), grpcPort)
                .usePlaintext()
                .build();

        try {
            RpcServiceGrpc.RpcServiceBlockingStub stub = RpcServiceGrpc.newBlockingStub(channel);
            List<double[]> dataPoints = collectDataPoints(stub, tokenLengths, repeats);

            if (dataPoints.size() < 3) {
                log.warn("Only {} data points collected, need at least 3; skipping fit", dataPoints.size());
                return;
            }

            double[] coeffs = fitPolynomial(dataPoints);
            double c0 = Math.max(0, coeffs[0]);
            double c1 = coeffs[1];
            double c2 = coeffs[2];

            TaskInfo.updateCoefficients(c0, c1, c2);
            log.info("Prefill profiling complete: T(n) = {} + {}*n + {}*n² (ms)",
                    c0, c1, c2);

            logResiduals(dataPoints, c0, c1, c2);
        } finally {
            channel.shutdown().awaitTermination(2, TimeUnit.SECONDS);
        }
    }

    private WorkerStatus waitForPrefillWorker(long timeoutMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            Map<String, WorkerStatus> workers =
                    engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
            if (workers != null) {
                for (WorkerStatus w : workers.values()) {
                    if (w != null && w.isAlive() && w.getIp() != null && w.getPort() > 0) {
                        return w;
                    }
                }
            }

            Map<String, WorkerStatus> pdfusionWorkers =
                    engineWorkerStatus.selectModelWorkerStatus(RoleType.PDFUSION, null);
            if (pdfusionWorkers != null) {
                for (WorkerStatus w : pdfusionWorkers.values()) {
                    if (w != null && w.isAlive() && w.getIp() != null && w.getPort() > 0) {
                        return w;
                    }
                }
            }
            Thread.sleep(1000);
        }
        return null;
    }

    private List<double[]> collectDataPoints(RpcServiceGrpc.RpcServiceBlockingStub stub,
                                              int[] tokenLengths, int repeats) {
        Random rng = new Random(42);
        List<double[]> points = new ArrayList<>();
        long requestIdBase = -900_000_000L;

        for (int tokenLen : tokenLengths) {
            for (int r = 0; r < repeats; r++) {
                try {
                    long timeUs = sendProfilingRequest(stub, rng, tokenLen,
                            requestIdBase - points.size());
                    if (timeUs > 0) {
                        points.add(new double[]{tokenLen, timeUs / 1000.0});
                        log.debug("Profile point: tokens={}, time={}ms", tokenLen, timeUs / 1000.0);
                    }
                } catch (Exception e) {
                    log.warn("Profiling request failed: tokens={}, repeat={}", tokenLen, r, e);
                }
            }
        }
        return points;
    }

    private long sendProfilingRequest(RpcServiceGrpc.RpcServiceBlockingStub stub,
                                      Random rng, int tokenLen, long requestId) {
        EngineRpcService.GenerateInputPB.Builder inputBuilder =
                EngineRpcService.GenerateInputPB.newBuilder()
                        .setRequestId(requestId)
                        .setIsFakeQuery(true);

        for (int i = 0; i < tokenLen; i++) {
            inputBuilder.addTokenIds(rng.nextInt(30000) + 1);
        }

        EngineRpcService.GenerateConfigPB.Builder configBuilder = inputBuilder.getGenerateConfigBuilder();
        configBuilder.setMaxNewTokens(1);
        configBuilder.setCanUsePdSeparation(false);
        configBuilder.setReuseCache(false);

        long startUs = System.nanoTime() / 1000;

        Iterator<EngineRpcService.GenerateOutputsPB> iter = stub
                .withDeadlineAfter(30, TimeUnit.SECONDS)
                .generateStreamCall(inputBuilder.build());

        long firstTokenCostUs = 0;
        while (iter.hasNext()) {
            EngineRpcService.GenerateOutputsPB output = iter.next();
            if (output.hasFlattenOutput()) {
                EngineRpcService.FlattenOutputPB flat = output.getFlattenOutput();
                if (flat.getAuxInfoCount() > 0) {
                    int costUs = flat.getAuxInfo(0).getFirstTokenCostTimeUs();
                    if (costUs > 0) {
                        firstTokenCostUs = costUs;
                    }
                }
            }
        }

        if (firstTokenCostUs > 0) {
            return firstTokenCostUs;
        }
        return System.nanoTime() / 1000 - startUs;
    }

    static double[] fitPolynomial(List<double[]> dataPoints) {
        int n = dataPoints.size();
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = dataPoints.get(i)[0];
            y[i] = dataPoints.get(i)[1];
        }

        // A^T * A for Vandermonde [1, x, x²]
        double s0 = n;
        double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        double b0 = 0, b1 = 0, b2 = 0;
        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double xi2 = xi * xi;
            s1 += xi;
            s2 += xi2;
            s3 += xi2 * xi;
            s4 += xi2 * xi2;
            b0 += y[i];
            b1 += y[i] * xi;
            b2 += y[i] * xi2;
        }

        // Solve 3x3 system via Cramer's rule:
        // [s0  s1  s2] [c0]   [b0]
        // [s1  s2  s3] [c1] = [b1]
        // [s2  s3  s4] [c2]   [b2]
        double det = det3(s0, s1, s2, s1, s2, s3, s2, s3, s4);
        if (Math.abs(det) < 1e-15) {
            return new double[]{b0 / n, 0, 0};
        }
        double c0 = det3(b0, s1, s2, b1, s2, s3, b2, s3, s4) / det;
        double c1 = det3(s0, b0, s2, s1, b1, s3, s2, b2, s4) / det;
        double c2 = det3(s0, s1, b0, s1, s2, b1, s2, s3, b2) / det;

        return new double[]{c0, c1, c2};
    }

    private static double det3(double a11, double a12, double a13,
                                double a21, double a22, double a23,
                                double a31, double a32, double a33) {
        return a11 * (a22 * a33 - a23 * a32)
             - a12 * (a21 * a33 - a23 * a31)
             + a13 * (a21 * a32 - a22 * a31);
    }

    private void logResiduals(List<double[]> dataPoints, double c0, double c1, double c2) {
        double sumSqErr = 0;
        for (double[] pt : dataPoints) {
            double predicted = c0 + c1 * pt[0] + c2 * pt[0] * pt[0];
            double err = pt[1] - predicted;
            sumSqErr += err * err;
            log.debug("  tokens={}, actual={}ms, predicted={}ms, err={}ms",
                    (int) pt[0], pt[1], predicted, err);
        }
        double rmse = Math.sqrt(sumSqErr / dataPoints.size());
        log.info("Prefill profiling RMSE: {}ms over {} points", rmse, dataPoints.size());
    }

    static int[] parseTokenLengths(String csv) {
        if (csv == null || csv.isBlank()) {
            return new int[]{32, 64, 128, 256, 512, 1024, 2048};
        }
        return Arrays.stream(csv.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .mapToInt(Integer::parseInt)
                .toArray();
    }
}
