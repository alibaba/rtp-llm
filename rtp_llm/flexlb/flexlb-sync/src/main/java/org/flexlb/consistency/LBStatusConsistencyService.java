package org.flexlb.consistency;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.domain.consistency.LBConsistencyConfig;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.util.JsonUtils;
import org.springframework.stereotype.Component;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
public class LBStatusConsistencyService implements MasterElectService {

    public static final String MASTER_CHANGE_NOTIFY_PATH = "/rtp_llm/notify_master";
    public static final ScheduledExecutorService SCHEDULED_EXECUTOR_SERVICE = new ScheduledThreadPoolExecutor(
            4,
            new NamedThreadFactory("LBStatusConsistencyService-Schedule-Thread"),
            new ThreadPoolExecutor.AbortPolicy()
    );

    private final ZookeeperMasterElectService zookeeperMasterElectService;
    private LBConsistencyConfig lbConsistencyConfig;
    private String hostIp;
    private String serverPort;
    private String roleId;

    public LBStatusConsistencyService(ZookeeperMasterElectService zookeeperMasterElectService) {
        this.zookeeperMasterElectService = zookeeperMasterElectService;
        this.init();
    }

    public void init() {
        log.info("start init LBStatusConsistencyService.");
        try {
            hostIp = InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            throw new RuntimeException(e);
        }
        serverPort = System.getProperty("server.port", "7001");
        log.info("hostIp:{}, serverPort:{}.", hostIp, serverPort);
        roleId = System.getenv("HIPPO_ROLE");
        if (StringUtils.isBlank(roleId)) {
            throw new RuntimeException("HIPPO_ROLE env is blank");
        }
        String configStr = System.getenv("WHALE_SYNC_LB_CONSISTENCY_CONFIG");
        log.info("WHALE_SYNC_LB_CONSISTENCY_CONFIG = {}.", configStr);
        if (configStr == null) {
            lbConsistencyConfig = new LBConsistencyConfig();
        } else {
            lbConsistencyConfig = JsonUtils.toObject(configStr, LBConsistencyConfig.class);
        }
        if (!lbConsistencyConfig.isNeedConsistency()) {
            log.warn("LBStatusConsistencyService is not need.");
            return;
        }
        log.info("start init ZookeeperMasterElectService.");

        SCHEDULED_EXECUTOR_SERVICE.scheduleWithFixedDelay(this::syncLBStatusFromMaster, 1000, 500, TimeUnit.MILLISECONDS);
    }

    @Override
    public void start() {
        if (!lbConsistencyConfig.isNeedConsistency()) {
            log.warn("start: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.start();
    }

    @Override
    public void offline() {
        if (!lbConsistencyConfig.isNeedConsistency()) {
            log.warn("offline: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.offline();
    }

    @Override
    public void destroy() {
        if (!lbConsistencyConfig.isNeedConsistency()) {
            log.warn("destroy: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.destroy();
    }

    @Override
    public boolean isMaster() {
        // 如果不需要一致性控制，则直接返回true，否则判断是否是master节点
        return !lbConsistencyConfig.isNeedConsistency() || zookeeperMasterElectService.isMaster();
    }

    @Override
    public String getMasterHostIp(boolean forceSync) {
        // 如果不需要一致性控制，则直接返回本机ip
        return lbConsistencyConfig.isNeedConsistency() ? zookeeperMasterElectService.getMasterHostIp(forceSync) : this.hostIp;
    }

    public String getMasterHostIpPort() {
        // 如果不需要一致性控制，则直接返回本机ip
        if (!lbConsistencyConfig.isNeedConsistency()) {
            return this.hostIp + ":" + serverPort;
        }
        return zookeeperMasterElectService.getMasterHostIp(false) + ":" + serverPort;
    }

    /**
     * 处理master变更
     *
     * @param req MasterChangeNotifyReq
     * @return MasterChangeNotifyResp
     */
    public MasterChangeNotifyResp handleMasterChange(MasterChangeNotifyReq req) {
        log.warn("recv MasterChangeNotifyReq:{}.", req);
        if (!roleId.equals(req.getRoleId())) {
            MasterChangeNotifyResp resp = new MasterChangeNotifyResp();
            resp.setSuccess(false);
            resp.setMsg("roleId not match this:" + roleId);
            return resp;
        }
        this.getMasterHostIp(true);
        MasterChangeNotifyResp resp = new MasterChangeNotifyResp();
        resp.setSuccess(true);
        return resp;
    }

    public SyncLBStatusResp dumpLBStatus() {
        SyncLBStatusResp resp = new SyncLBStatusResp();
        resp.setSuccess(true);
        // TODO 获取master status
        return resp;
    }

    /**
     * 从节点从主节点同步LB状态
     */
    private void syncLBStatusFromMaster() {
        // TODO 获取master status
    }
}
