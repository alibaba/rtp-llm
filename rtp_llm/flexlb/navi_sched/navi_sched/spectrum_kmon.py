from collections.abc import Mapping


def spectrum_kmon_tags(environ: Mapping[str, str]) -> dict[str, str]:
    """Identify the Spectrum application from the platform's own environment."""
    application = environ.get("SPECTRUM_APPLICATION_NAME", "").strip()
    workspace_id = environ.get("SPECTRUM_WORKSPACE_ID", "").strip()
    if not application or not workspace_id:
        return {}
    return {
        "hippo_app": application,
        "spectrum_application": application,
        "spectrum_workspace_id": workspace_id,
    }
