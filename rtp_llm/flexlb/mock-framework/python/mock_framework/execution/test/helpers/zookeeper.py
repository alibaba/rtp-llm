import os
from pathlib import Path


def command(root: Path, java):
    target = root / "java/zookeeper/target"
    classpath = os.pathsep.join(
        (str(target / "classes"), (target / "dependencies.txt").read_text().strip())
    )
    return [
        str(java),
        "-XX:ActiveProcessorCount=2",
        "-cp",
        classpath,
        "org.flexlb.mock.zookeeper.ZooKeeperMain",
    ]
