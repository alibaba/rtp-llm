package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class OptimizerStateInfo {

    @JsonProperty("full_location_spec_group_name")
    private String fullLocationSpecGroupName;

    @JsonProperty("linear_location_spec_group_name")
    private String linearLocationSpecGroupName;
}
