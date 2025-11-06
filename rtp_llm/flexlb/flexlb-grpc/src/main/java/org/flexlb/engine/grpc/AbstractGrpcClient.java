package org.flexlb.engine.grpc;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import io.grpc.ManagedChannel;
import io.grpc.stub.AbstractBlockingStub;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.util.CommonUtils;

/**
 * @author zjw
 * description:
 * date: 2025/4/23
 */
@Slf4j
public abstract class AbstractGrpcClient<STUB extends AbstractGrpcClient.GrpcStubWrapper> implements CustomNameResolver.Listener {

    /**
     * Maintain different channel for different service type.
     *
     * @see ServiceType
     */
    protected final Map<String/*ip:port:serviceType*/, Invoker> channelPool = new ConcurrentHashMap<>();
    protected final EngineLocalView engineLocalView;
    protected final GlobalCacheIndex globalCacheIndex;
    protected final GrpcReporter grpcReporter;

    protected AbstractGrpcClient(EngineLocalView engineLocalView,
                                 GlobalCacheIndex globalCacheIndex,
                                 GrpcReporter grpcReporter) {
        this.engineLocalView = engineLocalView;
        this.globalCacheIndex = globalCacheIndex;
        this.grpcReporter = grpcReporter;
    }

    @Override
    public void onAddressUpdate(List<String/*ip:port*/> hosts) {
        if (hosts == null) {
            log.error("received null hosts list");
            return;
        }
        log.warn("address update, size:{} currentSize:{}", hosts.size(), channelPool.size());

        Set<String/*ip:port:serviceType*/> currentKeys = new HashSet<>(channelPool.keySet());
        List<String/*ip:port:serviceType*/> addedKeys = new ArrayList<>();

        for (String host : hosts) {
            String[] parts = host.split(":");
            String ip = parts[0];
            int httpPort = Integer.parseInt(parts[1]);
            int grpcPort = CommonUtils.toGrpcPort(httpPort);

            String workerStatusKey = createKey(ip, grpcPort, ServiceType.WORKER_STATUS);
            String cacheStatusKey = createKey(ip, grpcPort, ServiceType.CACHE_STATUS);
            String multimodalWorkerStatusKey = createKey(ip, grpcPort, ServiceType.MULTIMODAL_WORKER_STATUS);
            String multimodalCacheStatusKey = createKey(ip, grpcPort, ServiceType.MULTIMODAL_CACHE_STATUS);
            boolean contained = currentKeys.remove(workerStatusKey) && currentKeys.remove(cacheStatusKey) 
                && currentKeys.remove(multimodalWorkerStatusKey) && currentKeys.remove(multimodalCacheStatusKey);

            if (!contained) {
                addedKeys.add(workerStatusKey);
                addedKeys.add(cacheStatusKey);
                addedKeys.add(multimodalWorkerStatusKey);
                addedKeys.add(multimodalCacheStatusKey);
            }
        }

        // add to pool
        for (String newKey : addedKeys) {
            if (!channelPool.containsKey(newKey)) {
                try {
                    ManagedChannel managedChannel = createChannel(newKey);
                    channelPool.put(newKey, new Invoker(newKey, managedChannel));
                    log.info("add channel for host {}", newKey);
                } catch (Exception e) {
                    log.error("create channel for host {} failed", newKey, e);
                }
            }
        }

        // remove and shutdown not alive invoker
        for (String key : currentKeys) {
            Invoker invoker = channelPool.remove(key);
            if (invoker != null) {
                try {
                    invoker.shutdown();
                } catch (Exception e) {
                    log.error("shutdown channel for host {} failed", invoker.getChannelKey(), e);
                }
            }

            String[] parsedKey = parseServiceKey(key);
            String ipPort = parsedKey[0] + ":" + parsedKey[1];
            engineLocalView.removeAllCacheBlockOfEngine(ipPort);
            globalCacheIndex.removeAllCacheBlockOfEngine(ipPort);
        }

        grpcReporter.reportChannelPoolSize(channelPool.size());
    }

    protected Invoker getInvoker(String channelKey) {
        Invoker invoker = channelPool.get(channelKey);
        if (invoker == null) {
            log.warn("ip:{} grpc channel not found, channelPool:{}", channelKey, channelPool);
        }
        return invoker;
    }

    protected abstract ManagedChannel createChannel(String hostKey);

    protected abstract STUB createStub(ManagedChannel channel);

    protected static String createKey(String ip, int port, ServiceType serviceType) {
        return ip + ":" + port + ":" + serviceType.getSuffix();
    }

    protected static String[] parseServiceKey(String serviceKey) {
        String[] parts = serviceKey.split(":");
        if (parts.length == 3) {
            return new String[]{parts[0], parts[1], parts[2]};
        }

        throw new IllegalArgumentException("Invalid service key format: " + serviceKey);
    }

    /**
     * Wrapper class for different gRPC service stubs
     */
    public static class GrpcStubWrapper {
        private final RpcServiceGrpc.RpcServiceBlockingStub rpcServiceStub;
        private final MultimodalRpcServiceGrpc.MultimodalRpcServiceBlockingStub multimodalRpcServiceStub;

        public GrpcStubWrapper(RpcServiceGrpc.RpcServiceBlockingStub rpcServiceStub,
                               MultimodalRpcServiceGrpc.MultimodalRpcServiceBlockingStub multimodalRpcServiceStub) {
            this.rpcServiceStub = rpcServiceStub;
            this.multimodalRpcServiceStub = multimodalRpcServiceStub;
        }

        public RpcServiceGrpc.RpcServiceBlockingStub getRpcServiceStub() {
            return rpcServiceStub;
        }

        public MultimodalRpcServiceGrpc.MultimodalRpcServiceBlockingStub getMultimodalRpcServiceStub() {
            return multimodalRpcServiceStub;
        }

        public GrpcStubWrapper withDeadlineAfter(long timeout, TimeUnit unit) {
            return new GrpcStubWrapper(
                    rpcServiceStub.withDeadlineAfter(timeout, unit),
                    multimodalRpcServiceStub.withDeadlineAfter(timeout, unit)
            );
        }
    }

    @Getter
    public class Invoker {

        private final String channelKey;
        private final ManagedChannel channel;
        private final GrpcStubWrapper rpcServiceStub;

        public Invoker(String channelKey, ManagedChannel channel) {
            this.channelKey = channelKey;
            this.channel = channel;
            this.rpcServiceStub = createStub(channel);
        }

        public void shutdown() {
            if (channel != null) {
                channel.shutdown();
            }
        }
    }

    /**
     * gRPC service type, in order to mark different gRPC connection in channelPool
     *
     * @see AbstractGrpcClient#channelPool
     * @see EngineGrpcClient
     */
    public enum ServiceType {

        WORKER_STATUS("worker", "GetWorkerStatus"),
        CACHE_STATUS("cache", "GetCacheStatus"),
        MULTIMODAL_WORKER_STATUS("multimodal_worker", "GetWorkerStatus"),
        MULTIMODAL_CACHE_STATUS("multimodal_cache", "GetCacheStatus");

        @Getter
        private final String suffix;
        @Getter
        private final String operationName;

        ServiceType(String suffix, String operationName) {
            this.suffix = suffix;
            this.operationName = operationName;
        }
    }
}
