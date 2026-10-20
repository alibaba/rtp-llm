"""Package a separately pinned schema-3 Master into the test-only bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def build(output, internal, spec_path):
    spec = json.loads(spec_path.read_text())
    sha = spec["source_commit"]
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Master must be pinned by a full commit SHA")
    if spec["schema_version"] != 3 or spec["jar"] != "master.jar":
        raise ValueError("pinned Master requires schema 3 and master.jar")

    def run(args, cwd):
        subprocess.run(args, cwd=cwd, check=True)

    if subprocess.run(["git", "cat-file", "-e", sha + "^{commit}"], cwd=REPO,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        run(["git", "fetch", "--depth=1", "origin", sha], REPO)
    with tempfile.TemporaryDirectory(prefix="pinned-master-") as temporary:
        work = Path(temporary)
        archive = work / "master.tar"
        with archive.open("wb") as stream:
            subprocess.run(["git", "archive", sha, "rtp_llm/flexlb", "rtp_llm/cpp/model_rpc/proto"],
                           cwd=REPO, stdout=stream, check=True)
        run(["tar", "xf", str(archive), "-C", str(work)], REPO)
        project = work / "rtp_llm/flexlb"
        # Build-profile addition only. Production Java and protobuf sources
        # remain byte-for-byte from the pinned release. Native local discovery
        # consumes the bundle's discovery_file without a source adapter.
        pom = project / "flexlb-api/pom.xml"
        text = pom.read_text()
        if text.count("<profiles>") != 1 or "<id>whale-bundle</id>" in text:
            raise ValueError("unexpected pinned Master Maven profile layout")
        text = text.replace("<profiles>", """<profiles>
          <profile><id>whale-bundle</id><dependencies><dependency>
            <groupId>org.flexlb</groupId><artifactId>kmonitor</artifactId>
          </dependency></dependencies></profile>""", 1)
        pom.write_text(text)
        run(["mvn", "-B", "-Popensource,!internal", "-pl", "flexlb-common", "-am", "install", "-DskipTests"], project)
        run(["mvn", "-B", "-pl", "kmonitor", "-am", "install", "-DskipTests"], internal / "java")
        run(["mvn", "-B", "-Popensource,!internal,whale-bundle", "-pl", "flexlb-api", "-am", "package", "-DskipTests"], project)
        output.mkdir(parents=True, exist_ok=True)
        target = output / spec["jar"]
        target.write_bytes((project / "flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar").read_bytes())
        spec["jar_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        spec["packaging"] = "pinned production Java and proto, opensource discovery, internal KMonitor"
        (output / "pinned-master-build.json").write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--internal-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=ROOT / "pinned-master-spec.json")
    args = parser.parse_args()
    build(args.output.resolve(), args.internal_root.resolve(), args.spec.resolve())
