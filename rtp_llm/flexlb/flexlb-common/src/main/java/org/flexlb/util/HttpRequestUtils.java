package org.flexlb.util;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.constant.HttpHeaderNames;
import org.flexlb.dao.BalanceContext;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiConsumer;

@Slf4j
public class HttpRequestUtils {

    // 预定义的头部处理器映射，避免运行时字符串比较
    public static final Map<String, BiConsumer<BalanceContext, String>> HEADER_PROCESSORS = new HashMap<>();

    static {

        HEADER_PROCESSORS.put(HttpHeaderNames.TRACE_PARENT.toLowerCase(), BalanceContext::setOtlpTraceParent);
        HEADER_PROCESSORS.put(HttpHeaderNames.TRACE_STATE.toLowerCase(), BalanceContext::setOtlpTraceState);
    }
}
