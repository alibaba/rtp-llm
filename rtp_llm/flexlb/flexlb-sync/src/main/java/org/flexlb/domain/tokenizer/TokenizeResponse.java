package org.flexlb.domain.tokenizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class TokenizeResponse {

    @JsonProperty("token_ids")
    private int[] tokenIds;
}
