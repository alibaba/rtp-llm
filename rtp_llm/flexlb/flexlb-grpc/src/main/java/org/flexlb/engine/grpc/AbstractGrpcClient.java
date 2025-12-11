package org.flexlb.engine.grpc;

import io.grpc.ManagedChannel;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.CustomNameResolver;
import org.flexlb.util.CommonUtils;
import org.springframework.scheduling.annotation.Scheduled;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author zjw
 * description:
 * date: 2025/4/23
 */
@Slf4j
public abstract class AbstractGrpcClient<STUB> implements CustomNameResolver.Listener {

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

    /**
     * 处理服务地址更新事件
     * 当服务发现检测到 worker 列表变化时，同步更新 gRPC channel 池和缓存视图
     *
     * @param ipPortList 最新的 worker 地址列表，格式为 ip:httpPort
     */
    @Override
    public void onAddressUpdate(List<String/*ip:port*/> ipPortList) {
        if (ipPortList == null) {
            log.error("received null ipPort list");
            return;
        }

        // 更新 gRPC channel 池
        updateGrpcChannelPool(ipPortList);

        // 更新引擎缓存，清理已下线的 engine
        updateEngineKvCache(ipPortList);
    }

    /**
     * 根据最新的 ipPortList 列表更新 gRPC channel 池
     * 新增 channel 连接新上线的 worker，移除已下线的 worker 的 channel
     *
     * @param ipPortList 最新的 worker 地址列表，格式为 ip:httpPort
     */
    private void updateGrpcChannelPool(List<String> ipPortList) {
        log.warn("address update, size:{} currentSize:{}", ipPortList.size(), channelPool.size());

        Set<String/*ip:port:serviceType*/> currentKeys = new HashSet<>(channelPool.keySet());
        List<String/*ip:port:serviceType*/> addedKeys = new ArrayList<>();

        // 识别新增和保留的 worker，标记需要移除的 channel
        for (String ipPort : ipPortList) {
            String[] parts = ipPort.split(":");
            String ip = parts[0];
            int httpPort = Integer.parseInt(parts[1]);
            int grpcPort = CommonUtils.toGrpcPort(httpPort);

            String workerStatusKey = createKey(ip, grpcPort, ServiceType.WORKER_STATUS);
            String cacheStatusKey = createKey(ip, grpcPort, ServiceType.CACHE_STATUS);
            boolean contained = currentKeys.remove(workerStatusKey) && currentKeys.remove(cacheStatusKey);

            if (!contained) {
                addedKeys.add(workerStatusKey);
                addedKeys.add(cacheStatusKey);
            }
        }

        // 为新上线的 worker 创建 channel
        for (String newKey : addedKeys) {
            if (!channelPool.containsKey(newKey)) {
                try {
                    ManagedChannel managedChannel = createChannel(newKey);
                    channelPool.put(newKey, new Invoker(newKey, managedChannel));
                    log.info("add channel for ipPort {}", newKey);
                } catch (Exception e) {
                    log.error("create channel for ipPort {} failed", newKey, e);
                }
            }
        }

        // 关闭并移除已下线 worker 的 channel
        for (String key : currentKeys) {
            Invoker invoker = channelPool.remove(key);
            if (invoker != null) {
                try {
                    invoker.shutdown();
                } catch (Exception e) {
                    log.error("shutdown channel for ipPort {} failed", invoker.getChannelKey(), e);
                }
            }
        }
    }

    /**
     * 更新缓存，清理已下线的 engine 缓存
     *
     * @param ipPortList 最新的 worker 地址列表，格式为 ip:httpPort
     */
    private void updateEngineKvCache(List<String> ipPortList) {
        Set<String> cacheEngineKeys = engineLocalView.getAllEngineIpPorts();
        Set<String> newEngineIpPorts = new HashSet<>(ipPortList);

        // size 相同时跳过
        if (cacheEngineKeys.size() == newEngineIpPorts.size()) {
            return;
        }

        // 找出需要清理的已下线 engine
        Set<String> staleEngineKeys = new HashSet<>(cacheEngineKeys);
        staleEngineKeys.removeAll(newEngineIpPorts);

        if (CollectionUtils.isNotEmpty(staleEngineKeys)) {
            log.info("Update cache: found {} stale engines to remove, current cache size: {}, new ipPortList size: {}",
                    staleEngineKeys.size(), cacheEngineKeys.size(), newEngineIpPorts.size());

            for (String staleEngine : staleEngineKeys) {
                log.warn("Removing stale engine cache: {}", staleEngine);
                long startTime = System.nanoTime() / 1000;
                engineLocalView.removeAllCacheBlockOfEngine(staleEngine);
                globalCacheIndex.removeAllCacheBlockOfEngine(staleEngine);
                long elapsed = System.nanoTime() / 1000 - startTime;
                log.warn("Removed stale engine cache: {} in {}μs", staleEngine, elapsed);
            }
        }
    }

    @Scheduled(fixedRate = 2000)
    public void reportChannelPoolSize() {
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

    @Getter
    public class Invoker {

        private final String channelKey;
        private final ManagedChannel channel;
        private final STUB rpcServiceStub;
        private final long createTime;
        private volatile long lastUsedTime;
        private volatile long expireTime;

        public Invoker(String channelKey, ManagedChannel channel) {
            this.channelKey = channelKey;
            this.channel = channel;
            this.rpcServiceStub = createStub(channel);
            long currentTime = System.nanoTime() / 1000;
            this.createTime = currentTime;
            this.lastUsedTime = currentTime;
            this.expireTime = 0;
        }

        public void updateLastUsedTime() {
            this.lastUsedTime = System.nanoTime() / 1000;
        }

        public void markExpired() {
            this.expireTime = System.nanoTime() / 1000;
        }

        public long getConnectionDuration() {
            return expireTime > 0 ? expireTime - createTime : System.nanoTime() / 1000 - createTime;
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
        CACHE_STATUS("cache", "GetCacheStatus");

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
