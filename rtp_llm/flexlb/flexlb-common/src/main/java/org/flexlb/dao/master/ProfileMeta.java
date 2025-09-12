package org.flexlb.dao.master;


import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.Map;

@Data
@Slf4j
public class ProfileMeta {
    @JsonProperty("profile_time")
    private Map<Integer, Integer> profileTime; // {token_size, time}
}
