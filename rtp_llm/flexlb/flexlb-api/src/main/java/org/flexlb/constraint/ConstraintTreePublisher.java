package org.flexlb.constraint;

import org.flexlb.constraint.ConstraintTreeModels.PublicationResult;
import org.flexlb.constraint.ConstraintTreeModels.SerializedArtifact;

public interface ConstraintTreePublisher {

    default ConstraintTreeModels.PreparedBuild prepare(ConstraintTreeModels.BuildRequest request) {
        return new ConstraintTreeModels.PreparedBuild(request, "");
    }

    PublicationResult publish(SerializedArtifact artifact);
}
