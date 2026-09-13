"""Build pinned production master sources with test-only discovery packaging.

No production Java sources are edited. KMonitor stays enabled; excluding the
VipServer dependency selects the existing DOMAIN_ADDRESS test discovery adapter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def run(args, cwd):
    subprocess.run(args, cwd=cwd, check=True)


def build(output, internal):
    spec = json.loads((ROOT / "legacy-master.json").read_text())
    # CI checks out a shallow tree. rev-parse of a literal SHA does not prove
    # the object exists; fetch that exact revision before archiving it.
    revision = spec["source_commit"] + "^{commit}"
    if subprocess.run(["git", "cat-file", "-e", revision], cwd=REPO,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        run(["git", "fetch", "--depth=1", "origin", spec["source_commit"]], REPO)
    sha = subprocess.check_output(["git", "rev-parse", "--verify", revision], cwd=REPO, text=True).strip()
    if sha != spec["source_commit"]:
        raise ValueError("legacy master source must be pinned by full SHA")
    with tempfile.TemporaryDirectory(prefix="legacy-master-") as directory:
        work = Path(directory)
        archive = work / "source.tar"
        with archive.open("wb") as f:
            subprocess.run(["git", "archive", sha, "rtp_llm/flexlb", "rtp_llm/cpp/model_rpc/proto"], cwd=REPO, stdout=f, check=True)
        run(["tar", "xf", str(archive), "-C", str(work)], REPO)
        project = work / "rtp_llm/flexlb"
        pom = project / "flexlb-api/pom.xml"
        text = pom.read_text()
        text = text.replace("<profiles>", """<profiles>
          <profile><id>whale-bundle</id><dependencies><dependency>
            <groupId>org.flexlb</groupId><artifactId>kmonitor</artifactId>
          </dependency></dependencies></profile>""", 1)
        pom.write_text(text)
        tests = project / "flexlb-common/src/test/java/org/flexlb/config"
        (tests / "WhaleLegacyConfigTest.java").write_text("""package org.flexlb.config;
import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.assertEquals;
class WhaleLegacyConfigTest {
 @Test void productionDocumentParsesWithoutTranslation() throws Exception {
  var config = ConfigService.parse(Files.readString(Path.of("inner-master.json")));
  assertEquals(1, config.getSchemaVersion());
  assertEquals(NonBatchDispatcherConfig.class, config.getDispatcher().getClass());
 }
}
""")
        shutil.copyfile(ROOT / "glm53-inner-master.json", project / "flexlb-common/inner-master.json")
        run(["mvn", "-B", "-Popensource,!internal", "-pl", "flexlb-common", "-am", "install",
             "-Dtest=WhaleLegacyConfigTest", "-Dsurefire.failIfNoSpecifiedTests=false"], project)
        run(["mvn", "-B", "-pl", "kmonitor", "-am", "install", "-DskipTests"], internal / "java")
        run(["mvn", "-B", "-Popensource,!internal,whale-bundle", "-pl", "flexlb-api", "-am", "package",
             "-Dtest=WhaleLegacyConfigTest", "-Dsurefire.failIfNoSpecifiedTests=false"], project)
        output.mkdir(parents=True, exist_ok=True)
        target = output / spec["jar"]
        shutil.copyfile(project / "flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar", target)
        spec["jar_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        spec["packaging"] = "opensource plus KMonitor; production Java sources unchanged"
        (output / "legacy-master-build.json").write_text(json.dumps(spec, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--internal-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve(), args.internal_root.resolve())
