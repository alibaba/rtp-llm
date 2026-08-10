package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Data;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Encapsulates all optimizer instance registration params and provides remote consistency check.
 */
@Data
@Builder
@JsonIgnoreProperties(ignoreUnknown = true)
public class OptimizerInstanceParams {

    @JsonProperty("block_size")
    private int blockSize;

    @JsonProperty("location_spec_infos")
    private List<OptimizerRegisterRequest.LocationSpecInfo> locationSpecInfos;

    @JsonProperty("location_spec_groups")
    private List<OptimizerRegisterRequest.LocationSpecGroup> locationSpecGroups;

    @JsonProperty("linear_step")
    private int linearStep;

    @JsonProperty("full_group_name")
    private String fullGroupName;

    @JsonProperty("instance_group")
    private String instanceGroup;

    /**
     * Check whether local params match the remote getInstance response (including instanceGroup).
     */
    public boolean matchesRemote(OptimizerGetInstanceResponse remote) {
        if (remote == null) return false;

        if (!Objects.equals(nullToEmpty(this.instanceGroup), nullToEmpty(remote.getInstanceGroup()))) return false;
        if (this.blockSize != remote.getBlockSize()) return false;
        if (this.linearStep != remote.getLinearStep()) return false;
        if (!Objects.equals(nullToEmpty(this.fullGroupName), nullToEmpty(remote.getFullGroupName()))) return false;
        if (!matchLocationSpecInfos(remote.getLocationSpecInfos())) return false;
        if (!matchLocationSpecGroups(remote.getLocationSpecGroups())) return false;

        return true;
    }

    private boolean matchLocationSpecInfos(List<OptimizerGetInstanceResponse.LocationSpecInfo> remoteInfos) {
        List<OptimizerRegisterRequest.LocationSpecInfo> localInfos = this.locationSpecInfos;
        if (localInfos == null || localInfos.isEmpty()) {
            return remoteInfos == null || remoteInfos.isEmpty();
        }
        if (remoteInfos == null || remoteInfos.size() != localInfos.size()) return false;

        // null-safe sort to prevent NPE when name is null
        Comparator<String> nameCmp = Comparator.nullsFirst(String::compareTo);
        List<OptimizerRegisterRequest.LocationSpecInfo> sortedLocal = new ArrayList<>(localInfos);
        sortedLocal.sort(Comparator.comparing(OptimizerRegisterRequest.LocationSpecInfo::getName, nameCmp));
        List<OptimizerGetInstanceResponse.LocationSpecInfo> sortedRemote = new ArrayList<>(remoteInfos);
        sortedRemote.sort(Comparator.comparing(OptimizerGetInstanceResponse.LocationSpecInfo::getName, nameCmp));

        for (int i = 0; i < sortedLocal.size(); i++) {
            OptimizerRegisterRequest.LocationSpecInfo local = sortedLocal.get(i);
            OptimizerGetInstanceResponse.LocationSpecInfo remote = sortedRemote.get(i);
            if (!Objects.equals(local.getName(), remote.getName()) || local.getSize() != remote.getSize()) {
                return false;
            }
        }
        return true;
    }

    private boolean matchLocationSpecGroups(List<OptimizerGetInstanceResponse.LocationSpecGroup> remoteGroups) {
        List<OptimizerRegisterRequest.LocationSpecGroup> localGroups = this.locationSpecGroups;
        if (localGroups == null || localGroups.isEmpty()) {
            return remoteGroups == null || remoteGroups.isEmpty();
        }
        if (remoteGroups == null || remoteGroups.size() != localGroups.size()) return false;

        // null-safe sort to prevent NPE when name is null
        Comparator<String> nameCmp = Comparator.nullsFirst(String::compareTo);
        List<OptimizerRegisterRequest.LocationSpecGroup> sortedLocal = new ArrayList<>(localGroups);
        sortedLocal.sort(Comparator.comparing(OptimizerRegisterRequest.LocationSpecGroup::getName, nameCmp));
        List<OptimizerGetInstanceResponse.LocationSpecGroup> sortedRemote = new ArrayList<>(remoteGroups);
        sortedRemote.sort(Comparator.comparing(OptimizerGetInstanceResponse.LocationSpecGroup::getName, nameCmp));

        for (int i = 0; i < sortedLocal.size(); i++) {
            OptimizerRegisterRequest.LocationSpecGroup local = sortedLocal.get(i);
            OptimizerGetInstanceResponse.LocationSpecGroup remote = sortedRemote.get(i);
            if (!Objects.equals(local.getName(), remote.getName())) return false;
            List<String> localSpecs = local.getSpecNames();
            List<String> remoteSpecs = remote.getSpecNames();
            if (localSpecs == null || localSpecs.isEmpty()) {
                if (remoteSpecs != null && !remoteSpecs.isEmpty()) return false;
            } else if (remoteSpecs == null || localSpecs.size() != remoteSpecs.size()) {
                return false;
            } else {
                // null-safe sort to prevent NPE when specName is null
                Comparator<String> specCmp = Comparator.nullsFirst(String::compareTo);
                List<String> sortedLocalSpecs = new ArrayList<>(localSpecs);
                sortedLocalSpecs.sort(specCmp);
                List<String> sortedRemoteSpecs = new ArrayList<>(remoteSpecs);
                sortedRemoteSpecs.sort(specCmp);
                if (!Objects.equals(sortedLocalSpecs, sortedRemoteSpecs)) return false;
            }
        }
        return true;
    }

    private static String nullToEmpty(String s) {
        return s == null ? "" : s;
    }
}
