package org.flexlb.util;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.constant.HttpHeaderNames;
import org.flexlb.dao.BalanceContext;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiConsumer;

@Slf4j
public class HttpRequestUtils {

    // Predefined header processor mapping to avoid runtime string comparison
    public static final Map<String, BiConsumer<BalanceContext, String>> HEADER_PROCESSORS = new HashMap<>();

    static {

        HEADER_PROCESSORS.put(HttpHeaderNames.TRACE_PARENT.toLowerCase(), BalanceContext::setOtlpTraceParent);
        HEADER_PROCESSORS.put(HttpHeaderNames.TRACE_STATE.toLowerCase(), BalanceContext::setOtlpTraceState);
    }
}
