package org.flexlb.mock.engine.vit;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Selects the OSS endpoint for a ViT image download.
 */
record VitDownloadRouting(Set<String> internalRegions, boolean rewriteOss) {
    private static final String REGION = "[a-z0-9]+(?:-[a-z0-9]+)+";
    private static final String OSS_SUFFIX = ".aliyuncs.com";
    private static final Pattern OSS_HOST = Pattern.compile(
        "((?:[a-z0-9-]+\\.)*)oss-(" + REGION + "?)(-internal)?\\.aliyuncs\\.com\\.?");
    private static final Set<String> GROUP_ENDPOINTS = Set.of("cn-zhangjiakou", "cn-shanghai",
        "cn-hangzhou", "cn-wulanchabu", "cn-beijing", "cn-shenzhen", "ap-southeast-1");

    static VitDownloadRouting load(JsonNode config) throws IOException {
        JsonNode regions = config.get("oss_internal_region_allowlist");
        if (regions != null && regions.isTextual()) {
            regions = new ObjectMapper().readTree(regions.textValue());
        }
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
        return new VitDownloadRouting(internalRegions, !config.path("enable_oss_internal_rewrite").asText().equals("0"));
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
}
