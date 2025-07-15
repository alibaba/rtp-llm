import os


def get_environments() -> dict[str, str]:
    map = dict()
    site = os.getenv("SIGMA_APP_SITE")
    if site is not None:
        map["site"] = site
    unit = os.getenv("SIGMA_APP_UNIT")
    if unit is not None:
        map["unit"] = unit
    app = os.getenv("SIGMA_APP_NAME")
    if app is not None:
        map["app"] = app
    stage = os.getenv("SIGMA_APP_STAGE")
    if stage is not None:
        map["stage"] = stage
    labels = os.getenv("NACOS_ENV_LABELS")
    if labels is not None:
        label_array = labels.split(",")

        for label in label_array:
            kv = label.split(":")
            if len(kv) == 2:
                map[kv[0].strip()] = kv[1].strip()

    return map
