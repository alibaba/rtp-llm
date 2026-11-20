package org.flexlb.cache;

import com.fasterxml.jackson.annotation.JsonProperty;

public record CacheMatchFailoverRequest(@JsonProperty("action") CacheMatchFailoverAction action) {}
