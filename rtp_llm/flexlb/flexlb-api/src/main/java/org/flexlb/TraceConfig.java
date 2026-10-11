package org.flexlb;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.exc.MismatchedInputException;
import com.fasterxml.jackson.databind.exc.UnrecognizedPropertyException;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.net.URI;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.cert.CertificateFactory;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** 与 Python config.py 共享 JSON 配置契约。 */
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY,
        setterVisibility = JsonAutoDetect.Visibility.NONE)
final class TraceConfig {

    static final String ENV = "RTP_LLM_TRACE_CONFIG";

    private static final Set<String> FIELDS = Set.of("enabled", "sampler_ratio", "endpoint", "headers",
            "certificate",
            "max_queue_size", "max_export_batch_size", "schedule_delay_ms", "http_timeout_ms");

    /** 需要下界（> 0）与 int 范围检查的字段。 */
    private static final List<String> INT_FIELDS = List.of("max_queue_size", "max_export_batch_size",
            "schedule_delay_ms", "http_timeout_ms");

    /** 禁止被 headers 覆盖的 HTTP 头。 */
    private static final Set<String> RESERVED = Set.of("host", "content-length", "content-type", "connection",
            "transfer-encoding", "content-encoding", "trailer", "upgrade", "keep-alive", "te");

    private static final ObjectMapper MAPPER = JsonMapper.builder()
            .propertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
            .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
            .build();

    private Boolean enabled;
    private Double samplerRatio;
    private String endpoint;
    private Map<String, String> headers;
    private String certificate;
    private Integer maxQueueSize;
    private Integer maxExportBatchSize;
    private Integer scheduleDelayMs;
    private Integer httpTimeoutMs;
    private String source;

    public boolean enabled() {
        return Boolean.TRUE.equals(enabled);
    }

    public double samplerRatio() {
        return samplerRatio == null ? 1.0 : samplerRatio;
    }

    public String endpoint() {
        return endpoint == null ? "" : endpoint;
    }

    public Map<String, String> headers() {
        return headers == null ? Map.of() : headers;
    }

    public String certificate() {
        return certificate == null ? "" : certificate;
    }

    public int maxQueueSize() {
        return maxQueueSize == null ? 2048 : maxQueueSize;
    }

    public int maxExportBatchSize() {
        return maxExportBatchSize == null ? 512 : maxExportBatchSize;
    }

    public int scheduleDelayMs() {
        return scheduleDelayMs == null ? 5000 : scheduleDelayMs;
    }

    public int httpTimeoutMs() {
        return httpTimeoutMs == null ? 3000 : httpTimeoutMs;
    }

    public String source() {
        return source;
    }

    static final class ConfigException extends IllegalArgumentException {
        final String field;
        final String code;

        ConfigException(String field, String code) {
            super(field + ":" + code);
            this.field = field;
            this.code = code;
        }
    }

    @Override
    public String toString() {
        // 不输出 headers：其中可能含接收端凭证。
        return "TraceConfig(enabled=" + enabled() + ", source=" + source + ")";
    }

    static TraceConfig disabled() {
        TraceConfig config = new TraceConfig();
        config.source = "disabled";
        return config;
    }

    static TraceConfig parse(String raw) {
        if (raw == null || raw.isBlank()) {
            return disabled();
        }
        JsonNode root = decode(raw);
        validateFieldNames(root);
        Boolean enabled = lenientEnabled(root);
        if (!enabled) {
            // 关闭时忽略其余字段：不反序列化、不做业务校验。
            return disabled();
        }
        normalizeScalars((ObjectNode) root, enabled);
        validateHeadersShape(root);
        validateScalarNulls(root);
        TraceConfig config = map(root);
        validate(config);
        return config;
    }

    /** 安全解析：不附加原始异常，避免 Jackson 消息带出 JSON 内容与凭证。 */
    private static JsonNode decode(String raw) {
        JsonNode node;
        try {
            node = MAPPER.readTree(raw);
        } catch (com.fasterxml.jackson.core.JsonParseException error) {
            String code = error.getOriginalMessage().startsWith("Duplicate field") ? "duplicate_key" : "invalid_json";
            throw new ConfigException("config", code);
        } catch (IOException ignored) {
            throw new ConfigException("config", "invalid_json");
        }
        if (node == null || !node.isObject()) {
            throw new ConfigException("config", "expected_object");
        }
        return node;
    }

    private static void validateFieldNames(JsonNode root) {
        Iterator<String> names = root.fieldNames();
        while (names.hasNext()) {
            if (!FIELDS.contains(names.next())) {
                throw new ConfigException("config", "unknown_field");
            }
        }
    }

    /** Normalize accepted scalar spellings before binding so Jackson defaults cannot change the contract. */
    private static void normalizeScalars(ObjectNode root, boolean enabled) {
        root.put("enabled", enabled);
        for (String name : INT_FIELDS) {
            JsonNode node = root.get(name);
            if (node == null || node.isNull()) {
                continue;
            }
            String value = node.isTextual() ? node.textValue().strip() : node.asText();
            if ((!node.isNumber() && !node.isTextual())
                    || (node.isTextual() && !value.matches("[+-]?[0-9]+"))) {
                throw new ConfigException(name, "expected_integer");
            }
            BigDecimal number;
            try {
                number = new BigDecimal(value).setScale(0, java.math.RoundingMode.DOWN);
            } catch (NumberFormatException error) {
                throw new ConfigException(name, "out_of_range");
            }
            if (number.signum() <= 0 || number.compareTo(BigDecimal.valueOf(Integer.MAX_VALUE)) > 0) {
                throw new ConfigException(name, "out_of_range");
            }
            root.put(name, number.intValue());
        }
        JsonNode ratio = root.get("sampler_ratio");
        if (ratio != null && !ratio.isNull()) {
            if (!ratio.isNumber() && !ratio.isTextual()) {
                throw new ConfigException("sampler_ratio", "expected_number");
            }
            String value = ratio.isTextual() ? ratio.textValue().strip() : ratio.asText();
            if (ratio.isTextual() && !value.matches("[+-]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?")) {
                throw new ConfigException("sampler_ratio", "expected_number");
            }
            double number;
            try {
                number = Double.parseDouble(value);
            } catch (NumberFormatException error) {
                throw new ConfigException("sampler_ratio", "expected_number");
            }
            if (!Double.isFinite(number) || number < 0 || number > 1) {
                throw new ConfigException("sampler_ratio", "out_of_range");
            }
            root.put("sampler_ratio", number);
        }
        for (String name : List.of("endpoint", "certificate")) {
            JsonNode node = root.get(name);
            if (node != null && !node.isNull() && node.isValueNode()) {
                root.put(name, node.asText().strip());
            }
        }
    }

    /** headers 必须是对象，且每个值必须是字符串（Map&lt;String, String&gt; 会被 Jackson 放宽）。 */
    private static void validateHeadersShape(JsonNode root) {
        JsonNode node = root.get("headers");
        if (node == null) {
            return;
        }
        if (!node.isObject()) {
            throw new ConfigException("headers", "expected_object");
        }
        node.fields().forEachRemaining(entry -> {
            if (!entry.getValue().isTextual()) {
                throw new ConfigException("headers", "invalid_header");
            }
        });
    }

    private static void validateScalarNulls(JsonNode root) {
        rejectExplicitNull(root, "sampler_ratio", "expected_number");
        for (String name : INT_FIELDS) {
            rejectExplicitNull(root, name, "expected_integer");
        }
        rejectExplicitNull(root, "endpoint", "expected_string");
        rejectExplicitNull(root, "certificate", "expected_string");
    }

    private static void rejectExplicitNull(JsonNode root, String name, String code) {
        JsonNode node = root.get(name);
        if (node != null && node.isNull()) {
            throw new ConfigException(name, code);
        }
    }

    /** 缺省关闭；显式 null 按配置错误处理。 */
    private static Boolean lenientEnabled(JsonNode root) {
        JsonNode node = root.get("enabled");
        if (node == null) {
            return Boolean.FALSE;
        }
        if (node.isBoolean()) {
            return node.booleanValue();
        }
        if (node.isIntegralNumber() && node.canConvertToInt()) {
            int value = node.intValue();
            if (value == 0) {
                return Boolean.FALSE;
            }
            if (value == 1) {
                return Boolean.TRUE;
            }
        }
        if (node.isTextual()) {
            String text = node.textValue().strip().toLowerCase(Locale.ROOT);
            if (text.equals("true") || text.equals("1")) {
                return Boolean.TRUE;
            }
            if (text.equals("false") || text.equals("0")) {
                return Boolean.FALSE;
            }
        }
        throw new ConfigException("enabled", "expected_boolean");
    }

    /** 直接反序列化；Jackson 的异常按字段与目标类型映射回契约错误码。 */
    private static TraceConfig map(JsonNode root) {
        try {
            TraceConfig config = MAPPER.treeToValue(root, TraceConfig.class);
            config.source = "manual";
            Map<String, String> normalized = new LinkedHashMap<>();
            validateHeaders(config.headers());
            config.headers().forEach((name, value) -> normalized.put(name.toLowerCase(Locale.ROOT), value));
            config.headers = Collections.unmodifiableMap(normalized);
            return config;
        } catch (ConfigException error) {
            throw error;
        } catch (UnrecognizedPropertyException error) {
            throw new ConfigException("config", "unknown_field");
        } catch (MismatchedInputException error) {
            throw mapError(error);
        } catch (IOException | RuntimeException error) {
            throw new ConfigException("config", "invalid_json");
        }
    }

    private static ConfigException mapError(MismatchedInputException error) {
        String field = error.getPath().isEmpty() ? "config" : error.getPath().get(0).getFieldName();
        if (!FIELDS.contains(field == null ? "" : field)) {
            field = "config";
        }
        Class<?> target = error.getTargetType() == null ? Object.class : error.getTargetType();
        String code;
        if (target == Double.class) {
            code = "expected_number";
        } else if (target == Boolean.class) {
            code = "expected_boolean";
        } else if (target == Integer.class) {
            code = "expected_integer";
        } else if (target == String.class) {
            code = "expected_string";
        } else {
            code = "invalid_json";
        }
        return new ConfigException(field, code);
    }

    private static void validate(TraceConfig config) {
        double ratio = config.samplerRatio();
        if (!Double.isFinite(ratio) || ratio < 0 || ratio > 1) {
            throw new ConfigException("sampler_ratio", "out_of_range");
        }
        if (config.maxQueueSize() <= 0) {
            throw new ConfigException("max_queue_size", "out_of_range");
        }
        if (config.maxExportBatchSize() <= 0) {
            throw new ConfigException("max_export_batch_size", "out_of_range");
        }
        if (config.scheduleDelayMs() <= 0) {
            throw new ConfigException("schedule_delay_ms", "out_of_range");
        }
        if (config.httpTimeoutMs() <= 0) {
            throw new ConfigException("http_timeout_ms", "out_of_range");
        }
        if (config.maxExportBatchSize() > config.maxQueueSize()) {
            throw new ConfigException("max_export_batch_size", "batch_exceeds_queue");
        }
        if (config.endpoint().isEmpty() || config.headers().isEmpty()) {
            throw new ConfigException("config", "incomplete_manual");
        }
        validateEndpoint(config.endpoint());
        validateCertificate(config.certificate());
    }

    private static void validateHeaders(Map<String, String> headers) {
        Map<String, String> seen = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : headers.entrySet()) {
            String name = entry.getKey();
            String value = entry.getValue();
            String normalized = name.toLowerCase(Locale.ROOT);
            if (!name.matches("[!#$%&'*+.^_`|~0-9A-Za-z-]+") || RESERVED.contains(normalized)
                    || value == null || value.isBlank() || !value.equals(value.strip())
                    || value.chars().anyMatch(c -> c < 32 && c != '\t' || c == 127 || c > 255)) {
                throw new ConfigException("headers", "invalid_header");
            }
            if (seen.putIfAbsent(normalized, value) != null) {
                throw new ConfigException("headers", "duplicate_header");
            }
        }
    }

    private static void validateCertificate(String certificate) {
        if (certificate.isEmpty()) {
            return;
        }
        try (var input = Files.newInputStream(Path.of(certificate))) {
            if (CertificateFactory.getInstance("X.509").generateCertificates(input).isEmpty()) {
                throw new ConfigException("certificate", "invalid_certificate");
            }
        } catch (Exception ignored) {
            throw new ConfigException("certificate", "invalid_certificate");
        }
    }

    private static void validateEndpoint(String endpoint) {
        try {
            URI uri = new URI(endpoint);
            String scheme = uri.getScheme();
            if (!("http".equalsIgnoreCase(scheme) || "https".equalsIgnoreCase(scheme))
                    || uri.getHost() == null || uri.getRawUserInfo() != null || uri.getRawFragment() != null
                    || uri.getPort() == 0 || uri.getPort() > 65535 || uri.getRawAuthority().endsWith(":")
                    || endpoint.chars().anyMatch(c -> c <= 32 || c >= 127)) {
                throw new IllegalArgumentException();
            }
        } catch (Exception ignored) {
            throw new ConfigException("endpoint", "invalid_endpoint");
        }
    }
}
