package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchIdGenerator;
import org.flexlb.balance.scheduler.DiagnosticsProvider;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.shadow.StateShadowBridge;
import org.flexlb.util.Logger;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

@Component
public class EndpointRegistry implements DiagnosticsProvider {

    private final ConcurrentHashMap<String, PrefillEndpoint> prefillEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, PrefillEndpoint> pdFusionEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, SimpleWorkerEndpoint> vitEndpoints = new ConcurrentHashMap<>();
    private final ConfigService configService;
    private final EngineGrpcClient grpcClient;
    private final BatchDispatchExecutor dispatchExecutor;
    private final BatchSchedulerReporter reporter;
    private final BatchIdGenerator batchIdGenerator;
    /** 状态账本门面（EP 记账与读点的数据源；关态 DISABLED 短路）。 */
    private final StateShadowBridge shadowBridge;

    public EndpointRegistry(ConfigService configService,
                            EngineGrpcClient grpcClient,
                            BatchDispatchExecutor dispatchExecutor,
                            BatchSchedulerReporter reporter,
                            Environment environment,
                            StateShadowBridge shadowBridge) {
        this.configService = configService;
        this.grpcClient = grpcClient;
        this.dispatchExecutor = dispatchExecutor;
        this.reporter = reporter;
        // null 归一为 DISABLED 单例：下游（ACK 回调的活跃计数 supplier、
        // endpoint 建造）恒非空，退化模式行为统一为 no-op。
        this.shadowBridge = shadowBridge != null ? shadowBridge : StateShadowBridge.DISABLED;
        // Initialize Snowflake batch ID generator with master identity;
        // one shared instance for all prefill endpoints.
        this.batchIdGenerator = new BatchIdGenerator(detectLocalIp(), detectPort(environment));
    }

    private static String detectLocalIp() {
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            Logger.warn("Failed to detect local IP, using 127.0.0.1 as fallback", e);
            return "127.0.0.1";
        }
    }

    private static int detectPort(Environment environment) {
        String portStr = environment == null ? null : environment.getProperty("server.port");
        if (portStr == null) {
            portStr = System.getProperty("server.port", "7001");
        }
        try {
            return Integer.parseInt(portStr);
        } catch (NumberFormatException e) {
            return 7001;
        }
    }

    public WorkerEndpoint get(RoleType roleType, String ipPort) {
        if (roleType == RoleType.PREFILL) {
            return getPrefill(ipPort);
        }
        if (roleType == RoleType.DECODE) {
            return getDecode(ipPort);
        }
        if (roleType == RoleType.PDFUSION) {
            return getPdFusion(ipPort);
        }
        if (roleType == RoleType.VIT) {
            return getVit(ipPort);
        }
        return null;
    }

    public Map<String, ? extends WorkerEndpoint> getEndpoints(RoleType roleType) {
        if (roleType == RoleType.PREFILL) {
            return prefillEndpoints;
        }
        if (roleType == RoleType.DECODE) {
            return decodeEndpoints;
        }
        if (roleType == RoleType.PDFUSION) {
            return pdFusionEndpoints;
        }
        if (roleType == RoleType.VIT) {
            return vitEndpoints;
        }
        return Map.of();
    }

    public PrefillEndpoint getPrefill(String ipPort) {
        return prefillEndpoints.get(ipPort);
    }

    public DecodeEndpoint getDecode(String ipPort) {
        return decodeEndpoints.get(ipPort);
    }

    private PrefillEndpoint getPdFusion(String ipPort) {
        return pdFusionEndpoints.get(ipPort);
    }

    private SimpleWorkerEndpoint getVit(String ipPort) {
        return vitEndpoints.get(ipPort);
    }

    public WorkerEndpoint ensureEndpoint(RoleType roleType, String ipPort, WorkerStatus status) {
        if (roleType == RoleType.PREFILL) {
            return ensurePrefillEndpoint(ipPort, status, roleType);
        }
        if (roleType == RoleType.DECODE) {
            return ensureDecodeEndpoint(ipPort, status);
        }
        if (roleType == RoleType.PDFUSION) {
            return ensurePdFusionEndpoint(ipPort, status, roleType);
        }
        if (roleType == RoleType.VIT) {
            return ensureVitEndpoint(ipPort, status);
        }
        throw new IllegalArgumentException("Unsupported role: " + roleType);
    }

    private PrefillEndpoint ensurePrefillEndpoint(String ipPort, WorkerStatus status, RoleType roleType) {
        PrefillEndpoint endpoint = prefillEndpoints.get(ipPort);
        if (endpoint != null && endpoint.getStatus() == status) {
            return endpoint;
        }
        return ensureEndpoint(prefillEndpoints, ipPort, status,
                candidateStatus -> createPrefillEndpoint(candidateStatus, roleType));
    }

    private DecodeEndpoint ensureDecodeEndpoint(String ipPort, WorkerStatus status) {
        return ensureEndpoint(decodeEndpoints, ipPort, status,
                this::createDecodeEndpoint);
    }

    private PrefillEndpoint ensurePdFusionEndpoint(String ipPort, WorkerStatus status, RoleType roleType) {
        PrefillEndpoint endpoint = pdFusionEndpoints.get(ipPort);
        if (endpoint != null && endpoint.getStatus() == status) {
            return endpoint;
        }
        return ensureEndpoint(pdFusionEndpoints, ipPort, status,
                candidateStatus -> createPrefillEndpoint(candidateStatus, roleType));
    }

    private SimpleWorkerEndpoint ensureVitEndpoint(String ipPort, WorkerStatus status) {
        return ensureEndpoint(vitEndpoints, ipPort, status,
                candidateStatus -> createSimpleEndpoint(candidateStatus, RoleType.VIT));
    }

    private <T extends WorkerEndpoint> T ensureEndpoint(ConcurrentHashMap<String, T> endpoints,
                                                         String ipPort,
                                                         WorkerStatus status,
                                                         Function<WorkerStatus, T> factory) {
        T current = endpoints.get(ipPort);
        if (current != null && current.getStatus() == status) {
            return current;
        }

        T candidate = factory.apply(status);
        while (true) {
            if (current == null) {
                T raced = endpoints.putIfAbsent(ipPort, candidate);
                if (raced == null) {
                    return candidate;
                }
                current = raced;
            }

            if (current.getStatus() == status) {
                candidate.close();
                return current;
            }
            if (endpoints.replace(ipPort, current, candidate)) {
                current.close();
                return candidate;
            }
            current = endpoints.get(ipPort);
        }
    }

    /**
     * Remove an endpoint only if it still belongs to the expired status generation.
     */
    public boolean remove(RoleType roleType, String ipPort, WorkerStatus expectedStatus) {
        if (expectedStatus == null) {
            return false;
        }
        expectedStatus.setAlive(false);
        if (roleType == RoleType.PREFILL) {
            return remove(prefillEndpoints, ipPort, expectedStatus);
        }
        if (roleType == RoleType.DECODE) {
            return remove(decodeEndpoints, ipPort, expectedStatus);
        }
        if (roleType == RoleType.PDFUSION) {
            return remove(pdFusionEndpoints, ipPort, expectedStatus);
        }
        if (roleType == RoleType.VIT) {
            return remove(vitEndpoints, ipPort, expectedStatus);
        }
        return false;
    }

    private <T extends WorkerEndpoint> boolean remove(ConcurrentHashMap<String, T> endpoints,
                                                       String ipPort,
                                                       WorkerStatus expectedStatus) {
        T endpoint = endpoints.get(ipPort);
        if (endpoint == null || endpoint.getStatus() != expectedStatus
                || !endpoints.remove(ipPort, endpoint)) {
            return false;
        }
        endpoint.close();
        return true;
    }

    private PrefillEndpoint createPrefillEndpoint(WorkerStatus status, RoleType roleType) {
        FlexlbConfig config = configService.loadBalanceConfig();
        prepareEndpointMetrics(roleType, status);
        // 账本门面仅注入 P/D 分离的 prefill 角色：PDFUSION 的引擎状态
        // 报文不进账本（事件泵对非 P/D 分离角色短路），若注入门面会让
        // P 条目绑定到永不推进的世代上（只能靠 janitor TTL 兑底清理）。
        StateShadowBridge bridgeForRole = roleType == RoleType.PREFILL ? shadowBridge : null;
        // 全局活跃计数（完成回显的队列长度口径）来自账本 P 侧快照；
        // 账本关（退化模式）无计数源，返回 0。
        java.util.function.IntSupplier globalActiveCount =
                () -> (int) shadowBridge.prefillActiveCountOrZero();
        return new PrefillEndpoint(status, config, grpcClient, dispatchExecutor,
                batchIdGenerator, globalActiveCount, reporter, bridgeForRole);
    }

    private DecodeEndpoint createDecodeEndpoint(WorkerStatus status) {
        prepareEndpointMetrics(RoleType.DECODE, status);
        return new DecodeEndpoint(status, configService.loadBalanceConfig(), shadowBridge);
    }

    private SimpleWorkerEndpoint createSimpleEndpoint(WorkerStatus status, RoleType roleType) {
        prepareEndpointMetrics(roleType, status);
        return new SimpleWorkerEndpoint(status);
    }

    private void prepareEndpointMetrics(RoleType roleType, WorkerStatus status) {
        reporter.prepareEndpointMetrics(roleType.name(), status.getIp());
    }

    public void close() {
        prefillEndpoints.values().forEach(WorkerEndpoint::close);
        decodeEndpoints.values().forEach(WorkerEndpoint::close);
        pdFusionEndpoints.values().forEach(WorkerEndpoint::close);
        vitEndpoints.values().forEach(WorkerEndpoint::close);
    }

    /**
     * Expose all prefill endpoints for per-worker metrics reporting.
     */
    public ConcurrentHashMap<String, PrefillEndpoint> getPrefillEndpoints() {
        return prefillEndpoints;
    }

    /**
     * Expose all decode endpoints for per-worker metrics reporting.
     */
    public ConcurrentHashMap<String, DecodeEndpoint> getDecodeEndpoints() {
        return decodeEndpoints;
    }

    public int getEndpointCount(RoleType roleType) {
        return getEndpoints(roleType).size();
    }

    /**
     * {@inheritDoc}
     *
     * <p>Returns the count of registered endpoints by role for the HTTP
     * diagnostic endpoints.
     */
    @Override
    public Map<String, Object> getDiagnostics() {
        Map<String, Object> diag = new LinkedHashMap<>();
        diag.put("prefill_endpoint_count", prefillEndpoints.size());
        diag.put("decode_endpoint_count", decodeEndpoints.size());
        diag.put("pd_fusion_endpoint_count", pdFusionEndpoints.size());
        diag.put("vit_endpoint_count", vitEndpoints.size());
        return diag;
    }
}
