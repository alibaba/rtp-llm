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


def build(output, internal, spec_path, master_config_path):
    spec = json.loads(spec_path.read_text())
    schema_version = spec.get("schema_version", 1)
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
        # Add an opt-in test Bean. Existing production Java files remain byte-for-byte unchanged.
        adapter = project / "flexlb-common/src/main/java/org/flexlb/mockdiscovery"
        adapter.mkdir(parents=True)
        shutil.copyfile(ROOT / "discovery_adapter/WhaleFileDiscovery.java", adapter / "WhaleFileDiscovery.java")
        adapter_tests = project / "flexlb-common/src/test/java/org/flexlb/mockdiscovery"
        adapter_tests.mkdir(parents=True)
        shutil.copyfile(ROOT / "discovery_adapter/WhaleFileDiscoveryTest.java", adapter_tests / "WhaleFileDiscoveryTest.java")

        if schema_version >= 1:
            # Schema-1 master: inject kmonitor profile and validate config parsing.
            pom = project / "flexlb-api/pom.xml"
            text = pom.read_text()
            text = text.replace("<profiles>", """<profiles>
          <profile><id>whale-bundle</id><dependencies><dependency>
            <groupId>org.flexlb</groupId><artifactId>kmonitor</artifactId>
          </dependency></dependencies></profile>""", 1)
            pom.write_text(text)
            tests = project / "flexlb-common/src/test/java/org/flexlb/config"
            dispatcher_type = json.loads(master_config_path.read_text())["dispatcher"]["type"]
            dispatcher_class = {
                "BATCH": "BatchDispatcherConfig",
                "NON_BATCH": "NonBatchDispatcherConfig",
            }.get(dispatcher_type)
            if dispatcher_class is None:
                raise ValueError("unsupported legacy dispatcher type: " + dispatcher_type)
            (tests / "WhaleLegacyConfigTest.java").write_text("""package org.flexlb.config;
import org.junit.jupiter.api.Test;
import java.nio.file.Files;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.assertEquals;
class WhaleLegacyConfigTest {
 @Test void productionDocumentParsesWithoutTranslation() throws Exception {
  var config = ConfigService.parse(Files.readString(Path.of("inner-master.json")));
  assertEquals(1, config.getSchemaVersion());
  assertEquals(DISPATCHER_CLASS.class, config.getDispatcher().getClass());
 }
}
""".replace("DISPATCHER_CLASS", dispatcher_class))
            shutil.copyfile(master_config_path, project / "flexlb-common/inner-master.json")
            run(["mvn", "-B", "-Popensource,!internal", "-pl", "flexlb-common", "-am", "install",
                 "-Dtest=WhaleLegacyConfigTest,WhaleFileDiscoveryTest", "-Dsurefire.failIfNoSpecifiedTests=false"], project)
            run(["mvn", "-B", "-pl", "kmonitor", "-am", "install", "-DskipTests"], internal / "java")
            run(["mvn", "-B", "-Popensource,!internal,whale-bundle", "-pl", "flexlb-api", "-am", "package",
                 "-Dtest=WhaleLegacyConfigTest,WhaleFileDiscoveryTest", "-Dsurefire.failIfNoSpecifiedTests=false"], project)
        else:
            # Schema-0 (pre-schema) master: flat FlexlbConfig, no kmonitor module.
            # Only inject WhaleFileDiscovery; skip config validation test.
            run(["mvn", "-B", "-Popensource,!internal", "-pl", "flexlb-common", "-am", "install",
                 "-Dtest=WhaleFileDiscoveryTest", "-Dsurefire.failIfNoSpecifiedTests=false"], project)
            run(["mvn", "-B", "-Popensource,!internal", "-pl", "flexlb-api", "-am", "package",
                 "-Dtest=WhaleFileDiscoveryTest", "-Dsurefire.failIfNoSpecifiedTests=false"], project)

        output.mkdir(parents=True, exist_ok=True)
        target = output / spec["jar"]
        shutil.copyfile(project / "flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar", target)
        spec["jar_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        spec["packaging"] = ("opensource plus KMonitor and opt-in WhaleFileDiscovery; original production Java sources unchanged"
                             if schema_version >= 1 else
                             "opensource plus opt-in WhaleFileDiscovery; flat config, no KMonitor; original production Java sources unchanged")
        (output / "legacy-master-build.json").write_text(json.dumps(spec, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--internal-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True,
                        help="External private source revision and jar specification")
    parser.add_argument("--master-config", type=Path, required=True,
                        help="External private schema 1 master configuration")
    args = parser.parse_args()
    build(args.output.resolve(), args.internal_root.resolve(),
          args.spec.resolve(), args.master_config.resolve())
