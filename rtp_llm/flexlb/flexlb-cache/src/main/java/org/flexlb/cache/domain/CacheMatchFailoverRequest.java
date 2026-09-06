package org.flexlb.cache.domain;

import com.fasterxml.jackson.annotation.JsonProperty;

public record CacheMatchFailoverRequest(@JsonProperty("action") CacheMatchFailoverAction action) {
}
