package org.flexlb.mock.engine.vit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Selects the proxy and OSS endpoint for a ViT image download.
 */
record VitDownloadRouting(Map<String, InetSocketAddress> proxies, Set<String> internalRegions, boolean rewriteOss) {
    private static final String REGION = "[a-z0-9]+(?:-[a-z0-9]+)+";
    private static final String OSS_SUFFIX = ".aliyuncs.com";
    private static final Pattern OSS_HOST = Pattern.compile(
        "((?:[a-z0-9-]+\\.)*)oss-(" + REGION + "?)(-internal)?\\.aliyuncs\\.com\\.?");
    private static final Set<String> GROUP_ENDPOINTS = Set.of("cn-zhangjiakou", "cn-shanghai",
        "cn-hangzhou", "cn-wulanchabu", "cn-beijing", "cn-shenzhen", "ap-southeast-1");

    static VitDownloadRouting load(JsonNode config) throws IOException {
        JsonNode map = jsonSetting(config, "intl_proxy_map");
        var proxies = new HashMap<String, InetSocketAddress>();
        if (map != null && !map.isNull()) {
            if (!map.isObject()) throw new IllegalArgumentException("INTL_PROXY_MAP must be a JSON object");
            map.fields().forEachRemaining(entry -> {
                if (!entry.getValue().isTextual()) throw new IllegalArgumentException("INTL_PROXY_MAP values must be strings");
                if (!entry.getValue().textValue().isEmpty()) proxies.put(entry.getKey(), proxyAddress(entry.getValue().textValue()));
            });
        }

        JsonNode regions = jsonSetting(config, "oss_internal_region_allowlist");
        Set<String> internalRegions = null;
        if (regions != null) {
            if (!regions.isArray()) throw new IllegalArgumentException("OSS_INTERNAL_REGION_ALLOWLIST must be a JSON array");
            var allowed = new HashSet<String>();
            for (JsonNode region : regions) {
                if (!region.isTextual() || !region.textValue().matches(REGION))
                    throw new IllegalArgumentException("Invalid OSS region in OSS_INTERNAL_REGION_ALLOWLIST");
                allowed.add(region.textValue());
            }
            internalRegions = Set.copyOf(allowed);
        }
        return new VitDownloadRouting(Map.copyOf(proxies), internalRegions,
            !config.path("enable_oss_internal_rewrite").asText().equals("0"));
    }

    /** Rewrites only the host, preserving the signed path and query verbatim. */
    URI endpoint(URI original) {
        if (!rewriteOss) return original;
        String host = original.getHost().toLowerCase(Locale.ROOT);
        var parts = OSS_HOST.matcher(host);
        if (!parts.matches()) return original;
        String region = parts.group(2);
        boolean internal = parts.group(3) != null;
        String target;
        if (internalRegions == null && !internal && GROUP_ENDPOINTS.contains(region)) {
            target = parts.group(1) + region + ".oss" + OSS_SUFFIX;
        } else if (internalRegions != null && internal && !internalRegions.contains(region)) {
            target = parts.group(1) + "oss-" + region + OSS_SUFFIX + (host.endsWith(".") ? "." : "");
        } else {
            return original;
        }
        String url = original.toString();
        int hostStart = original.getScheme().length() + 3;
        if (original.getRawUserInfo() != null) hostStart += original.getRawUserInfo().length() + 1;
        return URI.create(url.substring(0, hostStart) + target + url.substring(hostStart + host.length()));
    }

    private static JsonNode jsonSetting(JsonNode config, String key) throws IOException {
        JsonNode value = config.get(key);
        if (value != null && value.isTextual()) {
            return value.textValue().isBlank() && key.equals("intl_proxy_map")
                ? null : new ObjectMapper().readTree(value.textValue());
        }
        return value;
    }

    private static InetSocketAddress proxyAddress(String value) {
        URI proxy;
        try {
            proxy = URI.create(value);
        } catch (IllegalArgumentException error) {
            throw new IllegalArgumentException("Invalid HTTP proxy URL in INTL_PROXY_MAP");
        }
        if (!"http".equalsIgnoreCase(proxy.getScheme()) || proxy.getHost() == null
            || proxy.getRawUserInfo() != null || proxy.getRawQuery() != null || proxy.getRawFragment() != null
            || (!proxy.getRawPath().isEmpty() && !proxy.getRawPath().equals("/"))
            || proxy.getPort() == 0 || proxy.getPort() > 65535) {
            throw new IllegalArgumentException("INTL_PROXY_MAP requires an HTTP proxy URL without credentials");
        }
        return InetSocketAddress.createUnresolved(proxy.getHost(), proxy.getPort() == -1 ? 80 : proxy.getPort());
    }
}
