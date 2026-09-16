package org.flexlb.consistency;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.domain.consistency.LBConsistencyConfig;
import org.flexlb.domain.consistency.MasterChangeNotifyReq;
import org.flexlb.domain.consistency.MasterChangeNotifyResp;
import org.flexlb.domain.consistency.SyncLBStatusResp;
import org.flexlb.util.JsonUtils;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.LinkedHashMap;
import java.util.Map;

@Slf4j
@Component
public class LBStatusConsistencyService implements MasterElectService {

    public static final String MASTER_CHANGE_NOTIFY_PATH = "/rtp_llm/notify_master";

    private final ZookeeperMasterElectService zookeeperMasterElectService;
    private final Environment environment;
    private LBConsistencyConfig lbConsistencyConfig;
    private String serverPort;
    private String roleId;
    private String localHostIp;

    public LBStatusConsistencyService(ZookeeperMasterElectService zookeeperMasterElectService,
                                      Environment environment) {
        this.zookeeperMasterElectService = zookeeperMasterElectService;
        this.environment = environment;
        this.init();
    }

    public void init() {
        log.info("start init LBStatusConsistencyService.");
        try {
            localHostIp = InetAddress.getLocalHost().getHostAddress();
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
        roleId = System.getenv("HIPPO_ROLE");
        if (StringUtils.isBlank(roleId)) {
            throw new RuntimeException("HIPPO_ROLE env is blank");
        }
        String configStr = System.getenv("FLEXLB_SYNC_CONSISTENCY_CONFIG");
        log.info("FLEXLB_SYNC_CONSISTENCY_CONFIG = {}.", configStr);
        if (configStr == null) {
            lbConsistencyConfig = new LBConsistencyConfig();
        } else {
            lbConsistencyConfig = JsonUtils.toObject(configStr, LBConsistencyConfig.class);
        }
        if (!isNeedConsistency()) {
            log.warn("LBStatusConsistencyService is not need.");
            return;
        }
        log.info("start init ZookeeperMasterElectService.");

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
    public void destroy() {
        if (!isNeedConsistency()) {
            log.warn("destroy: lbConsistencyConfig is closed.");
            return;
        }
        this.zookeeperMasterElectService.destroy();
    }

    @Override
    public boolean isNeedConsistency() {
        return lbConsistencyConfig.isNeedConsistency();
    }

    @Override
    public boolean isMaster() {
        if (!isNeedConsistency()) {
            return false;
        }
        return zookeeperMasterElectService.isMaster();
    }

    @Override
    public void refreshMasterHost(boolean forceSync) {
        if (isNeedConsistency() && forceSync) {
            zookeeperMasterElectService.updateLatestMaster();
        }
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
        if (!roleId.equals(req.getRoleId())) {
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
}
