package org.flexlb.util;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.TypeReference;
import com.taobao.vipserver.client.core.VIPClient;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.constant.HttpHeaderNames;
import org.flexlb.dao.RequestContext;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.enums.LoadBalanceStrategyEnum;

import java.net.URI;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

@Slf4j
public class HttpRequestUtils {

    // 预定义的头部处理器映射，避免运行时字符串比较
    public static final Map<String, BiConsumer<RequestContext, String>> HEADER_PROCESSORS = new HashMap<>();

    static {

        HEADER_PROCESSORS.put(HttpHeaderNames.TRACE_PARENT.toLowerCase(), RequestContext::setOtlpTraceParent);
        HEADER_PROCESSORS.put(HttpHeaderNames.TRACE_STATE.toLowerCase(), RequestContext::setOtlpTraceState);
        HEADER_PROCESSORS.put(HttpHeaderNames.EAGLE_EYE_TRACE_ID.toLowerCase(), RequestContext::setOriginTraceId);
        HEADER_PROCESSORS.put(HttpHeaderNames.EAGLE_EYE_USER_DATA.toLowerCase(), RequestContext::setTraceUserData);
        HEADER_PROCESSORS.put(HttpHeaderNames.EAGLE_EYE_RPC_ID.toLowerCase(), RequestContext::setRpcId);
        
        // Use standard tracing headers as fallback
        HEADER_PROCESSORS.put(HttpHeaderNames.X_TRACE_ID.toLowerCase(), RequestContext::setOriginTraceId);
        HEADER_PROCESSORS.put(HttpHeaderNames.X_REQUEST_ID.toLowerCase(), RequestContext::setTraceUserData);
        HEADER_PROCESSORS.put(HttpHeaderNames.X_SPAN_ID.toLowerCase(), RequestContext::setRpcId);
        HEADER_PROCESSORS.put(HttpHeaderNames.CONTENT_LENGTH.toLowerCase(), (ctx, value) -> {
            try {
                Long contentLength = Long.parseLong(value);
                ctx.setReqContentLength(contentLength);
            } catch (NumberFormatException e) {
                log.error("parse content length error", e);
            }
        });
    }

    public static URI getUriByVipserver(String vipAddress) {
        try {
            String ipPort = VIPClient.srvIP(vipAddress);
            if (StringUtils.isBlank(ipPort)) {
                log.error("get vipserver ip port is null");
                return null;
            }
            String url = "http://" + ipPort;
            return new URI(url);
        } catch (Exception e) {
            log.error("get uri failed", e);
            return null;
        }
    }

    public static URI createURI(Endpoint endpoint) {
        if (endpoint == null) {
            log.error("get uri failed, endpoint is null");
            return null;
        }
        String type = endpoint.getType();
        String address = endpoint.getAddress();
        try {
            if (LoadBalanceStrategyEnum.VIPSERVER.getName().equals(type)) {
                return getUriByVipserver(address);
            } else if (LoadBalanceStrategyEnum.SPECIFIED_IP_PORT.getName().equals(type)) {
                return URI.create("http://" + address);
            }
            else if (LoadBalanceStrategyEnum.SPECIFIED_IP_PORT_LIST.getName().equals(type)) {
                List<String> ipPortList = JSON.parseObject(address, new TypeReference<List<String>>() {
                });
                // 随机选择一个
                String randomIpPort = ipPortList.get((int) (Math.random() * ipPortList.size()));
                return URI.create("http://" + randomIpPort);
            } else {
                log.error("get uri failed, endpoint type is not supported, type:{}", type);
                return null;
            }
        } catch (Exception e) {
            log.error("get uri failed", e);
            return null;
        }
    }
}
