package org.flexlb.metric;

import lombok.Getter;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * FlexMetricTags - Monitoring metric tags interface
 *
 * @author saichen.sm
 */
public interface FlexMetricTags {

    /**
     * Get tag key-value map
     *
     * @return Tag key-value map
     */
    Map<String, String> getTags();

    /**
     * Check if tags are empty
     *
     * @return true if empty, false otherwise
     */
    boolean isEmpty();

    /**
     * Immutable tags implementation class
     * Used to replace KMonitor's ImmutableMetricTags
     */
    class ImmutableFlexMetricTags implements FlexMetricTags {
        @Getter
        private final Map<String, String> tags;

        /**
         * Constructor - create from Map
         *
         * @param tags Tag map
         */
        public ImmutableFlexMetricTags(Map<String, String> tags) {
            if (tags == null || tags.isEmpty()) {
                this.tags = Collections.emptyMap();
            } else {
                this.tags = Collections.unmodifiableMap(new java.util.HashMap<>(tags));
            }
        }

        /**
         * Constructor - create from key-value pairs (supports multiple pairs)
         *
         * @param keyValues Key-value pairs, format: key1, value1, key2, value2, ...
         */
        public ImmutableFlexMetricTags(String... keyValues) {
            if (keyValues == null || keyValues.length == 0) {
                this.tags = Collections.emptyMap();
            } else if (keyValues.length % 2 != 0) {
                throw new IllegalArgumentException("Key-value pairs must be even number of arguments");
            } else {
                Map<String, String> tempTags = new HashMap<>();
                for (int i = 0; i < keyValues.length; i += 2) {
                    String key = keyValues[i];
                    String value = keyValues[i + 1];
                    if (key != null && value != null) {
                        tempTags.put(key, value);
                    }
                }
                this.tags = Collections.unmodifiableMap(tempTags);
            }
        }

        @Override
        public boolean isEmpty() {
            return tags.isEmpty();
        }

        @Override
        public String toString() {
            return "ImmutableFlexMetricTags{" + "tags=" + tags + '}';
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            ImmutableFlexMetricTags that = (ImmutableFlexMetricTags) o;
            return tags.equals(that.tags);
        }

        @Override
        public int hashCode() {
            return tags.hashCode();
        }
    }

    /**
     * Factory method - create immutable tags from Map
     *
     * @param tags Tag map
     * @return Immutable tags instance
     */
    static FlexMetricTags of(Map<String, String> tags) {
        return new ImmutableFlexMetricTags(tags);
    }

    /**
     * Factory method - create immutable tags from key-value pairs
     *
     * @param keyValues Key-value pairs, format: key1, value1, key2, value2, ...
     * @return Immutable tags instance
     */
    static FlexMetricTags of(String... keyValues) {
        return new ImmutableFlexMetricTags(keyValues);
    }

    /**
     * Factory method - create immutable tags with engineIp and engineIpPort pre-populated
     *
     * @param engineIp       Engine IP (pure IP, e.g. "10.0.0.1")
     * @param engineIpPort   Engine IP:Port (e.g. "10.0.0.1:8080")
     * @param extraKeyValues Additional key-value pairs, format: key1, value1, key2, value2, ...
     * @return Immutable tags instance
     */
    static FlexMetricTags ofEngine(String engineIp, String engineIpPort, String... extraKeyValues) {
        java.util.Map<String, String> map = new java.util.HashMap<>();
        if (engineIp != null) {
            map.put("engineIp", engineIp);
        }
        if (engineIpPort != null) {
            map.put("engineIpPort", engineIpPort);
        }
        if (extraKeyValues != null && extraKeyValues.length % 2 != 0) {
            throw new IllegalArgumentException("extraKeyValues must have even number of elements");
        }
        if (extraKeyValues != null) {
            for (int i = 0; i + 1 < extraKeyValues.length; i += 2) {
                String key = extraKeyValues[i];
                String value = extraKeyValues[i + 1];
                if (key != null && value != null) {
                    map.put(key, value);
                }
            }
        }
        return new ImmutableFlexMetricTags(map);
    }
}
