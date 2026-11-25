package org.flexlb.service.monitor;

import java.util.regex.Pattern;

final class TtftRequestIds {
    private static final Pattern PREFILL_PROXY_SUFFIX = Pattern.compile(
            "(#call[0-9a-fA-F]{32}(?:#attempt[0-9]+)?)(?:-p-[0-9a-fA-F]{8}(?:#call[0-9a-fA-F]{32}(?:#attempt[0-9]+)?)?)+$");

    private TtftRequestIds() {
    }

    /**
     * The P proxy appends a transport suffix to its inner call. The complete DashClient execution
     * ID before that suffix, including its call and retry number, identifies one dispatch.
     */
    static String dispatchId(String requestId) {
        var suffix = PREFILL_PROXY_SUFFIX.matcher(requestId);
        return suffix.find() ? requestId.substring(0, suffix.end(1)) : requestId;
    }
}
