"""Validate local launch recipes independently of process ownership."""

from pathlib import Path

from ..config.validation import duration, fields, text
from . import ExecutionError
from .roles.master import validate_master_launch
from .roles.prometheus import prepare_prometheus


def prepare_launch(raw, case, base):
    """Launch recipes are developer bindings, separate from case business configuration."""
    fields(
        raw, {"components", "prometheus", "zookeeper"}, {"components"}, where="launch"
    )
    if not isinstance(raw["components"], dict):
        raise ExecutionError("launch components must be a mapping")
    if "zookeeper" in raw["components"]:
        raise ExecutionError(
            "ZooKeeper belongs in the launch.zookeeper recipe, not components"
        )
    components = {}
    recipes = dict(raw["components"])
    if "zookeeper" in raw:
        recipes["zookeeper"] = raw["zookeeper"]
    for role, recipe in recipes.items():
        if role != "zookeeper" and role not in case.environment:
            raise ExecutionError(
                f"Launch recipe names an unconfigured component: {role}"
            )
        fields(
            recipe,
            {"command", "cwd", "env", "startup_timeout"},
            {"command"},
            where="launch." + role,
        )
        command = recipe["command"]
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(v, str) or not v for v in command)
        ):
            raise ExecutionError(
                "command must be a nonempty argv list; no shell expansion"
            )
        env = recipe.get("env", {})
        if not isinstance(env, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()
        ):
            raise ExecutionError("Launch env must contain string keys and values")
        address_env = (
            {"MOCK_BIND_HOST", "MOCK_ADVERTISE_HOST"}
            if role in {"prefill", "decode", "vit"}
            else set()
        )
        if any(k.startswith("MOCK_") and k not in address_env for k in env):
            raise ExecutionError("MOCK_* launch variables are assigned by the runner")
        startup = recipe.get("startup_timeout", "30s")
        duration(startup, positive=True)
        cwd = Path(base) / text(recipe.get("cwd", "."), "launch.cwd")
        if not cwd.is_dir():
            raise ExecutionError(f"Launch cwd is not a directory: {cwd}")
        deployment = case.environment[role]["deployment"] if role != "zookeeper" else {}
        if deployment.get("resources") or deployment.get("heap"):
            raise ExecutionError(
                "This local launcher does not yet enforce resources/heap; refusing to ignore them"
            )
        components[role] = {
            **recipe,
            "command": command,
            "env": env,
            "cwd": str(cwd.resolve()),
            "startup_timeout": startup,
        }
    zookeeper = components.pop("zookeeper", None)
    native_master = "MODEL_SERVICE_CONFIG" in case.environment.get("master", {}).get(
        "config", {}
    )
    if native_master != (zookeeper is not None):
        raise ExecutionError(
            "Native Master requires a zookeeper launch recipe; other components do not use it"
        )
    missing = set(case.environment) - set(components)
    if missing:
        raise ExecutionError(f"Missing component launch recipes: {sorted(missing)}")
    if "master" in components:
        validate_master_launch(
            case.environment["master"]["config"],
            components["master"]["env"],
            local_election=zookeeper is not None,
        )
    result = {"components": components}
    if zookeeper is not None:
        result["zookeeper"] = zookeeper
    if "prometheus" in raw:
        result["prometheus"] = prepare_prometheus(raw["prometheus"], base)
        if any(
            "prometheus" in value.get("config", {})
            for value in case.environment.values()
        ):
            raise ExecutionError(
                "Managed Prometheus cannot be combined with component prometheus configuration"
            )
    return result


def local_launch(case, java, prometheus, *, framework):
    """Bind a Case to locally built RTP, Client, Master and ZooKeeper artifacts."""
    import os

    framework, java = Path(framework).resolve(), Path(java).resolve()
    if not java.is_file() or not os.access(java, os.X_OK):
        raise ExecutionError("java must be an executable Java 21 binary")
    unsupported = set(case.environment) - {
        "master",
        "prefill",
        "decode",
        "vit",
        "client",
    }
    if unsupported:
        raise ExecutionError(
            f"Local launch does not implement roles: {sorted(unsupported)}"
        )

    def artifact(relative):
        path = framework / relative
        if not path.is_file():
            raise ExecutionError(
                f"Missing build artifact: {path}; run scripts/build-java.sh first"
            )
        return path

    def component_classpath(role):
        root = f"java/{role}/target"
        dependencies = (
            artifact(f"{root}/dependencies.txt").read_text(encoding="utf-8").strip()
        )
        if not dependencies or any(
            not Path(entry).is_absolute() or not Path(entry).exists()
            for entry in dependencies.split(os.pathsep)
        ):
            raise ExecutionError(
                f"{role} classpath has missing or nonabsolute entries; run scripts/build-java.sh first"
            )
        return os.pathsep.join([str(framework / root / "classes"), dependencies])

    prefix = [str(java), "-XX:ActiveProcessorCount=2"]
    commands = {}
    raw = {"prometheus": {"binary": str(Path(prometheus).resolve())}}
    if "master" in case.environment:
        config = case.environment["master"]["config"]
        if not {"FLEXLB_CONFIG", "MODEL_SERVICE_CONFIG"} <= config.keys():
            raise ExecutionError(
                "Local Master requires FLEXLB_CONFIG and MODEL_SERVICE_CONFIG"
            )
        artifact("java/master/target/classes/org/flexlb/MasterNode.class")
        artifact(
            "java/zookeeper/target/classes/org/flexlb/mock/zookeeper/ZookeeperNode.class"
        )
        classpath = component_classpath("master")
        # Expose native metrics locally.
        commands["master"] = prefix + [
            "-Dflexlb.monitor.metric-whitelist=flexlb_",
            "-cp",
            classpath,
            "org.flexlb.MasterNode",
        ]
        raw["zookeeper"] = {
            "command": prefix
            + [
                "-cp",
                component_classpath("zookeeper"),
                "org.flexlb.mock.zookeeper.ZookeeperNode",
            ],
            "startup_timeout": "60s",
        }
    if {"prefill", "decode", "vit"} & case.environment.keys():
        engine = prefix + [
            "-cp",
            str(
                artifact("java/engine/target/mock-engine-service-1.0-SNAPSHOT-all.jar")
            ),
            "org.flexlb.mock.engine.EngineMain",
        ]
        commands.update(prefill=engine, decode=engine, vit=engine)
    if "client" in case.environment:
        commands["client"] = prefix + [
            "-jar",
            str(artifact("java/client/target/mock-client-1.0-SNAPSHOT-all.jar")),
        ]
    raw["components"] = {
        role: {"command": list(commands[role]), "startup_timeout": "60s"}
        for role in case.environment
    }
    if "master" in raw["components"]:
        raw["components"]["master"]["startup_timeout"] = "120s"
    return prepare_launch(raw, case, framework)
