"""Compatibility entry point; storage lives in experiment_store, views in reporting."""
from reporting.experiment import (
    ArtifactRole as ArtifactRole, ArtifactEntry as ArtifactEntry,
    ViewDescriptor as ViewDescriptor, ExperimentWriter as ExperimentWriter,
    ExperimentReader as ExperimentReader, publish_run as publish_run, main as main,
)

if __name__ == '__main__':
    main()
