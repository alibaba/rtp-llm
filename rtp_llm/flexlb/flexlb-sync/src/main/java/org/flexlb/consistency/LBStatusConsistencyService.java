package org.flexlb.consistency;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConsistencyConfig;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.config.ZookeeperConsistencyConfig;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.util.JsonUtils;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * <p>Initialization prepares the configured election service. Online starts participation; offline
 * withdraws it, while master queries and notifications expose the current election result.
 * Destruction releases the service resources.
 */
@Slf4j
@Component
public class LBStatusConsistencyService implements MasterElectService {
    public static final String MASTER_CHANGE_NOTIFY_PATH = "/rtp_llm/notify_master";

    private final ZookeeperMasterElectService zookeeperMasterElectService;
    private final Environment environment;
    private final ConsistencyConfig consistencyConfig;
    private final DeploymentIdentity deploymentIdentity;

    private String serverPort;
    private String roleId;
    private String localHostIp;

    public LBStatusConsistencyService(
            ZookeeperMasterElectService zookeeperMasterElectService,
            Environment environment,
            ConfigService configService,
            DeploymentIdentity deploymentIdentity) {
        this.zookeeperMasterElectService = zookeeperMasterElectService;
        this.environment = environment;
        this.consistencyConfig = configService.loadBalanceConfig().getConsistency();
        this.deploymentIdentity = deploymentIdentity;
        this.init();
    }

    public void init() {
        log.info("start init LBStatusConsistencyService.");
        try {
            localHostIp =
                    isNeedConsistency()
                            ? zookeeperMasterElectService.getLocalIp()
                            : InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            throw new RuntimeException(e);
        }
        // Read from Spring Environment to respect --server.port= CLI args;
        // fall back to JVM system property.
        serverPort = environment.getProperty("server.port");
        if (serverPort == null) {
            serverPort = System.getProperty("server.port", "7001");
        }
        log.info("hostIp:{}, serverPort:{}.", localHostIp, serverPort);
        if (!isNeedConsistency()) {
            log.warn("LBStatusConsistencyService is not need.");
            return;
        }
        roleId = deploymentIdentity.getDeploymentId();
        log.info("start init ZookeeperMasterElectService.");
    }

    @Override
    public boolean isNeedConsistency() {
        return consistencyConfig instanceof ZookeeperConsistencyConfig;
    }

    @Override
    public void start() {
        if (!isNeedConsistency()) {
            log.warn("start: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.start();
    }

    @Override
    public void offline() {
        if (!isNeedConsistency()) {
            log.warn("offline: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.offline();
    }

    @Override
    @PreDestroy
    public void destroy() {
        if (!isNeedConsistency()) {
            log.warn("destroy: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.destroy();
    }

    public String getLocalHostIp() {
        return localHostIp;
    }

    /**
     * Handle master change
     *
     * @param req MasterChangeNotifyReq
     * @return MasterChangeNotifyResp
     */
    public MasterChangeNotifyResp handleMasterChange(MasterChangeNotifyReq req) {
        log.warn("recv MasterChangeNotifyReq:{}.", req);
        if (roleId == null || !Objects.equals(roleId, req.getRoleId())) {
            MasterChangeNotifyResp resp = new MasterChangeNotifyResp();
            resp.setSuccess(false);
            resp.setMsg("roleId not match this:" + roleId);
            return resp;
        }
        this.refreshMasterHost(true);
        MasterChangeNotifyResp resp = new MasterChangeNotifyResp();
        resp.setSuccess(true);
        return resp;
    }

    @Override
    public void refreshMasterHost(boolean forceSync) {
        if (isNeedConsistency() && forceSync) {
            zookeeperMasterElectService.updateLatestMaster();
        }
    }

    public SyncLBStatusResp dumpLBStatus() {
        SyncLBStatusResp resp = new SyncLBStatusResp();
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("consistency_enabled", isNeedConsistency());
        snapshot.put("master", isMaster());
        snapshot.put("local_host", localHostIp);
        snapshot.put("master_host", getMasterHostIpPort());
        snapshot.put("server_port", serverPort);
        resp.setSuccess(true);
        resp.setMsg("leadership snapshot");
        resp.setLbStatus(JsonUtils.toStringOrEmpty(snapshot));
        return resp;
    }

    @Override
    public boolean isMaster() {
        if (!isNeedConsistency()) {
            return false;
        }
        return zookeeperMasterElectService.isMaster();
    }

    public String getMasterHostIpPort() {
        if (!isNeedConsistency()) {
            return null;
        }
        String masterHostIp = zookeeperMasterElectService.getMasterHostIp(false);
        if (masterHostIp == null) {
            return null;
        }
        return masterHostIp + ":" + serverPort;
    }
}
