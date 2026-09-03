package org.flexlb.constraint;

import org.flexlb.constraint.ConstraintTreeModels.PublicationResult;
import org.flexlb.constraint.ConstraintTreeModels.SerializedArtifact;

public interface ConstraintTreePublisher {

    PublicationResult publish(SerializedArtifact artifact);
}
