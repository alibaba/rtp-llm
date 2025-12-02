package org.flexlb.consistency;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.StringUtils;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.recipes.leader.CancelLeadershipException;
import org.apache.curator.framework.recipes.leader.LeaderSelector;
import org.apache.curator.framework.recipes.leader.LeaderSelectorListener;
import org.apache.curator.framework.recipes.leader.Participant;
import org.apache.curator.framework.state.ConnectionState;
import org.apache.curator.retry.ExponentialBackoffRetry;
import org.apache.curator.utils.CloseableUtils;
import org.flexlb.domain.consistency.LBConsistencyConfig;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.LoggingUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.net.InetAddress;
import java.net.URI;
import java.net.UnknownHostException;
import java.time.Duration;
import java.util.Collection;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.consistency.LBStatusConsistencyService.MASTER_CHANGE_NOTIFY_PATH;

/**
 * @author zjw
 * description:
 * date: 2025/3/30
 */
@Component
public class ZookeeperMasterElectService implements MasterElectService, LeaderSelectorListener {

    private static final Logger LOGGER = LoggerFactory.getLogger("syncConsistencyLogger");

    private static final String MASTER_NAMESPACE = "whale-master";
    private static final String MASTER_LEADER_PATH = "/master_lb_leader/";
    @Setter
    private LBConsistencyConfig lbConsistencyConfig;
    private final GeneralHttpNettyService generalHttpNettyService;
    private final EngineHealthReporter engineHealthReporter;
    @Setter
    private String roleId;
    @Setter
    private String ip;
    @Setter
    private int port;
    @Setter(AccessLevel.PACKAGE)
    private CuratorFramework client;
    @Setter(AccessLevel.PACKAGE)
    private LeaderSelector leaderSelector;
    @Getter
    private volatile boolean isMaster;
    private volatile boolean stopCompleteForLeader;
    private volatile String masterHost;

    private final AtomicReference<CountDownLatch> leaderCloseLatchRef = new AtomicReference<>();

    public ZookeeperMasterElectService(GeneralHttpNettyService generalHttpNettyService,
                                       EngineHealthReporter engineHealthReporter) {

        LoggingUtils.warn("Initializing ZookeeperMasterElectService...");

        this.generalHttpNettyService = generalHttpNettyService;
        this.engineHealthReporter = engineHealthReporter;

        init();
    }

    public void init() {
        initializeLBConsistencyConfig();
        if (!lbConsistencyConfig.isNeedConsistency()) {
            LOGGER.warn("Consistency is not required for LBConsistencyConfig.");
            return;
        }
        initializeRoleId();
        initializeIpAndPort();
        initializeZookeeperClient();
        scheduleMasterUpdateTask();
    }

    private void initializeRoleId() {
        roleId = System.getenv("HIPPO_ROLE");
        if (StringUtils.isBlank(roleId)) {
            throw new RuntimeException("Environment variable HIPPO_ROLE is not set or is blank");
        }
    }

    private void initializeIpAndPort() {
        try {
            ip = InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            throw new RuntimeException("Failed to retrieve local host address", e);
        }
        port = Integer.parseInt(System.getProperty("server.port", "7001"));
    }

    private void initializeLBConsistencyConfig() {
        String configStr = System.getenv("WHALE_SYNC_LB_CONSISTENCY_CONFIG");
        LOGGER.warn("WHALE_SYNC_LB_CONSISTENCY_CONFIG = {}.", configStr);

        lbConsistencyConfig = configStr == null
                ? new LBConsistencyConfig()
                : JsonUtils.toObject(configStr, LBConsistencyConfig.class);
    }

    private void initializeZookeeperClient() {
        try {
            LBConsistencyConfig.ZookeeperConfig zookeeperConfig = lbConsistencyConfig.getZookeeperConfig();
            client = CuratorFrameworkFactory.builder()
                    .namespace(MASTER_NAMESPACE)
                    .connectString(zookeeperConfig.getZkHost())
                    .sessionTimeoutMs(zookeeperConfig.getZkTimeoutMs())
                    .connectionTimeoutMs(zookeeperConfig.getZkTimeoutMs())
                    .retryPolicy(new ExponentialBackoffRetry(1000, 3))
                    .build();
            client.start();
            leaderSelector = new LeaderSelector(client, MASTER_LEADER_PATH + roleId, this);
            leaderSelector.setId(ip);
            // 在主节点任务完成后，自动重新参与选举
            leaderSelector.autoRequeue();
        } catch (Exception e) {
            LOGGER.warn("Failed to initialize Zookeeper client and leader selector for roleId: {}, currentHost: {}", roleId, ip, e);
            closeClient();
            closeLeaderSelector();
            throw new RuntimeException("Initialization failed", e);
        }
    }

    private void scheduleMasterUpdateTask() {
        LBStatusConsistencyService.SCHEDULED_EXECUTOR_SERVICE.scheduleWithFixedDelay(
                this::updateLatestMaster, 0, 5, TimeUnit.SECONDS);
    }

    /**
     * 启动选举过程
     */
    @Override
    public void start() {
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} doStart start.", roleId, ip);
        // 调用 leaderSelector.start() 向 Zookeeper 注册。
        // Zookeeper 在指定路径下创建一个临时顺序节点
        leaderSelector.start();
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} doStart finished.", roleId, ip);
    }

    /**
     * 关闭选举选择器
     */
    @Override
    public void offline() {
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} offline start.", roleId, ip);

        // 设置停止标志为true，表示不再继续作为领导者
        stopCompleteForLeader = true;

        trySignalCloseLatch();
        closeLeaderSelector();
        trySignalCloseLatch();
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} offline finished.", roleId, ip);
    }

    @Override
    public String getMasterHostIp(boolean forceSync) {
        if (forceSync) {
            updateLatestMaster();
        }
        if (isMaster) {
            return ip;
        } else {
            return masterHost;
        }
    }

    /**
     * 当前节点已经被选为主节点时的回调方法。
     * 该方法需要阻塞，以保持主节点的身份，直到需要释放主节点的时候，才可以返回
     *
     * @param curatorFramework the client
     */
    @Override
    public void takeLeadership(CuratorFramework curatorFramework) {
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} takeLeadership", roleId, ip);
        reportMasterEvent("BECOME_MASTER");
        if (stopCompleteForLeader) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} closed already.", roleId, ip);
            // 释放领导权
            reportMasterEvent("RELEASE_MASTER");
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} released LeaderShip.", roleId, ip);
            return;
        }

        // 成为主节点
        isMaster = true;
        try {
            CountDownLatch countDownLatch = new CountDownLatch(1);
            leaderCloseLatchRef.set(countDownLatch);
            // 主动通知其他参与者当前节点已成为主节点
            activelyNotifyParticipants();

            // 当前线程阻塞，等待主节点关闭后，释放主节点
            while (!Thread.currentThread().isInterrupted()) {
                if (countDownLatch.await(1000, TimeUnit.MILLISECONDS)) {
                    break;
                }
            }
            reportMasterEvent("MASTER_CLOSED");
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} recv close latch signal.", roleId, ip);
        } catch (InterruptedException e) {
            reportMasterEvent("MASTER_INTERRUPTED");
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} is interrupted.", roleId, ip);
        } catch (Exception e) {
            reportMasterEvent("MASTER_ERROR");
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} takeLeadership error.", roleId, ip, e);
        } finally {
            leaderCloseLatchRef.set(null);
            isMaster = false;
        }
        // 释放领导权
        reportMasterEvent("RELEASE_MASTER");
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} released LeaderShip.", roleId, ip);
    }

    /**
     * 当连接状态发生变化时的回调方法。
     *
     * @param curatorFramework the client
     * @param connectionState  the new state
     */
    @Override
    public void stateChanged(CuratorFramework curatorFramework, ConnectionState connectionState) {
        LOGGER.warn("ZKMasterElector roleId:{} stateChanged:{}", roleId, connectionState);
        if (connectionState == ConnectionState.LOST || connectionState == ConnectionState.SUSPENDED) {
            throw new CancelLeadershipException();
        }
    }

    /**
     * 更新获取最新的主节点
     */
    private void updateLatestMaster() {
        synchronized (this) {
            try {
                String leaderId = leaderSelector.getLeader().getId();
                if (StringUtils.isNotBlank(leaderId)) {
                    if (!leaderId.equals(masterHost)) {
                        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} leaderId change from {} to {}.", roleId, ip,
                                masterHost, leaderId);
                    }
                    masterHost = leaderId;
                }
            } catch (Exception e) {
                LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} getLeaderID error.", roleId, ip, e);
            }
        }
        reportMasterNode();
    }

    /**
     * 主动通知其他参与者当前节点已成为主节点
     */
    private void activelyNotifyParticipants() {
        try {
            Collection<Participant> participants = leaderSelector.getParticipants();
            for (Participant participant : participants) {
                // 只通知非主节点
                if (!participant.isLeader() && ip.equals(participant.getId())) {
                    notifyParticipant(participant.getId());
                }
            }
        } catch (Exception e) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} activelyNotifyParticipants error.", roleId, ip, e);
        }
    }

    private void notifyParticipant(String participantIp) {
        try {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} notifyParticipant:{}", roleId, ip, participantIp);
            MasterChangeNotifyReq req = new MasterChangeNotifyReq();
            req.setReqIp(ip);
            req.setRoleId(roleId);
            URI uri = new URI("http://" + participantIp + ":" + port);
            Mono<MasterChangeNotifyResp> mono =
                    generalHttpNettyService.request(req, uri, MASTER_CHANGE_NOTIFY_PATH, MasterChangeNotifyResp.class);
            mono.timeout(Duration.ofMillis(1000))
                    .toFuture()
                    .whenComplete((masterChangeNotifyResp, throwable) ->
                            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} notifyParticipant resp:{}", roleId, ip,
                                    masterChangeNotifyResp, throwable));
        } catch (Exception e) {
            LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} notifyParticipant error.", roleId, ip, e);
        }
    }

    private void reportMasterNode() {
        try {
            if (masterHost != null) {
                engineHealthReporter.reportPrefillBalanceMasterNode(masterHost);
            }
        } catch (Exception e) {
            // ignore
        }
    }

    private void reportMasterEvent(String event) {
        try {
            engineHealthReporter.reportPrefillBalanceMasterEvent(event);
        } catch (Exception e) {
            // ignore
        }
    }

    private void closeClient() {
        if (client != null) {
            try {
                CloseableUtils.closeQuietly(client);
            } catch (Exception e) {
                LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} closeClient error.", roleId, ip, e);
            }
        }
    }

    private void closeLeaderSelector() {
        if (leaderSelector != null) {
            try {
                CloseableUtils.closeQuietly(leaderSelector);
            } catch (Exception e) {
                LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} closeLeaderSelector error.", roleId, ip, e);
            }
        }
    }

    private void trySignalCloseLatch() {
        CountDownLatch latch = leaderCloseLatchRef.get();
        if (latch != null) {
            latch.countDown();
        }
    }

    @Override
    public void destroy() {
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} destroy start.", roleId, ip);
        offline();
        closeClient();
        LOGGER.warn("ZKMasterElector roleId:{} currentHost:{} destroy finished.", roleId, ip);
    }
}