"""Replay one load on a compiled FlexLB scheduler, with per-request inputs held fixed."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
MODULE = HERE.parents[1]
PACKAGE = "org.flexlb.mockengine."
CLASSES = ["PrefillRecoveryReplay.java", "ServiceControl.java", "ReplayFeedback.java"]
PARAMS = dict(
    workers=30,
    decode=50,
    sessions=270,
    blocks=256,
    normal=240,
    peak=240,
    rotate=120,
    base=112,
    common=40,
    extension=14,
    cycle=16,
    pool=119580,
    fullCache="true",
    uniformRounds="true",
    cyclic="true",
    baseline=65,
    overload=90,
    recovery=240,
    privateTail=1,
    pulse=0,
    moreWork="true",
    extra=6,
    secondary="true",
)
CONFIG_TEST = "PendingPrefixPlacementTest"
BASE_REF = "e44a12a93"


def run(cmd, log=None):
    if log is None:
        subprocess.run(cmd, cwd=MODULE, check=True)
    else:
        print(log, flush=True)
        with log.open("w") as output:
            subprocess.run(
                cmd, cwd=MODULE, stdout=output, stderr=subprocess.STDOUT, check=True
            )


def build(args, out, jdk):
    mvn = os.getenv("MVN") or shutil.which("mvn") or str(MODULE / "mvnw")
    run(
        [
            mvn,
            *args.maven_arg,
            "-pl",
            "flexlb-mock-engine",
            "-am",
            "-Dsurefire.failIfNoSpecifiedTests=false",
            "-Dtest="
            + CONFIG_TEST
            + ",EngineQueueAdmissionTest,JavaMockEngineClusterTest,KvTotalReportAlignmentTest,DecodeReserveNetAllocationTest,ProductionCaliberDecodeTest,ReplayCacheFeedbackTest",
            "test",
        ],
        out / "build.log",
    )
    reports = MODULE / "flexlb-mock-engine/target/surefire-reports"
    xml = next(reports.glob("TEST-*." + CONFIG_TEST + ".xml"))
    cp = (
        ElementTree.parse(xml)
        .find("./properties/property[@name='surefire.test.class.path']")
        .get("value")
    )
    # A concurrent Maven build can replace an inner class after the replay JVM has loaded
    # its enclosing class. Freeze module bytecode once for every arm of this comparison.
    classpath = []
    compiled_modules = {}
    for entry in cp.split(os.pathsep):
        if not entry:
            continue
        path = Path(entry)
        if path.is_relative_to(MODULE) and path.parent.name == "target":
            local = out / "classes" / path.parent.parent.name / path.name
            shutil.copytree(path, local, dirs_exist_ok=True)
            compiled_modules[path] = local
            path = local
        classpath.append(str(path))
    mock_classes = compiled_modules[MODULE / "flexlb-mock-engine/target/classes"]
    engine = mock_classes / "org/flexlb/mockengine"
    for name in ("JavaMockEngineCluster", "JavaMockEngineCluster$CancelFaultKind"):
        if not (engine / (name + ".class")).is_file():
            raise RuntimeError(f"incomplete frozen mock-engine build: {name}")
    cp = os.pathsep.join(classpath)
    current = out / "driver"
    current.mkdir()
    run(
        [
            jdk / "javac",
            "-proc:none",
            "-cp",
            cp,
            "-d",
            current,
            *[HERE / f for f in CLASSES],
        ]
    )
    # The reactor supplies the dependencies of the current, ABI-compatible module.
    # Compile changed baseline sources in memory. No git checkout or worktree mutation.
    git = lambda *cmd: subprocess.check_output(
        ["git", *cmd], cwd=MODULE, text=True
    ).strip()
    sha = git("rev-parse", "--verify", args.ref + "^{commit}")
    changed = git(
        "diff", "--no-renames", "--diff-filter=MD", "--name-only", sha, "--", "."
    ).splitlines()
    sources = [
        name
        for name in changed
        if "/src/main/java/" in name
        and name.endswith(".java")
        and "/flexlb-mock-engine/" not in name
    ]
    pinned = out / "baseline"
    pinned.mkdir()
    if sources:
        lombok = next(
            f for f in cp.split(os.pathsep) if Path(f).name.startswith("lombok-")
        )
        run(
            [
                jdk / "java",
                HERE / "CompilePinnedSources.java",
                sha,
                cp,
                lombok,
                pinned,
                *sources,
            ],
            out / "baseline-sources.log",
        )
    (out / "versions.txt").write_text(
        "baseline "
        + sha
        + "\ncheckout "
        + git("rev-parse", "HEAD")
        + "\n"
        + git("status", "--short", "--", ".")
        + "\n"
    )
    driver_cp = os.pathsep.join((str(current), cp))
    base_cp = os.pathsep.join((str(pinned), driver_cp))
    candidate = out / "candidate"
    candidate.mkdir()
    # This checkout can contain unrelated scheduler experiments. Overlay only the named,
    # Maven-built production classes (including nested classes) on the same baseline.
    for group in (
        "endpoint/EngineQueueAdmission",
        "endpoint/PrefillEndpoint",
        "scheduler/PlacementAvailability",
        "scheduler/OrderedRequestQueue",
        "scheduler/PlacementCapacityWaitQueue",
        "scheduler/QueuedRequestScheduler",
        "scheduler/GlobalQueueEntry",
        "scheduler/AbstractRequestScheduler",
        "scheduler/RequestRoute",
        "scheduler/WorkerBatcher",
    ):
        prefix = Path("org/flexlb/balance") / group
        classes = (
            compiled_modules[MODULE / "flexlb-sync/target/classes"] / prefix.parent
        )
        dest = candidate / prefix.parent
        dest.mkdir(parents=True, exist_ok=True)
        for source in [
            classes / (prefix.name + ".class"),
            *classes.glob(prefix.name + "$*.class"),
        ]:
            shutil.copyfile(source, dest / source.name)
    return {"base": base_cp, "checkout": os.pathsep.join((str(candidate), base_cp))}


def write_geometry(source, dest):
    with gzip.open(source, "rt") as stream:
        rows = json.load(stream)["rows"]
    with dest.open("wb") as binary:
        binary.write(struct.pack(">i", len(rows)))
        for _, length, _, blocks in rows:
            binary.write(struct.pack(">ii", length, len(blocks)))
            binary.write(struct.pack(f">{len(blocks)}i", *blocks))


def replay(args):
    out = args.out or MODULE / "target/prefill-recovery" / time.strftime(
        "%Y%m%d-%H%M%S"
    )
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not os.environ.get("JAVA_HOME"):
        raise SystemExit("Set JAVA_HOME to the JDK 21 directory for this build.")
    jdk = Path(os.environ["JAVA_HOME"]) / "bin"
    cp = build(args, out, jdk)
    if args.build_only and not args.exit_control:
        return
    params = dict(
        PARAMS, config=args.config.resolve(), seed=args.seed, maxLagMs=args.max_lag_ms
    )

    def launch(version, name, extra=None, main="PrefillRecoveryReplay", rest=()):
        settings = (
            params | (extra or {}) | {"feedbackFile": out / (name + "-feedback.bin")}
        )
        run(
            [
                jdk / "java",
                "-ea",
                "-Xmx" + args.heap,
                "-XX:ActiveProcessorCount=6",
                "-XX:+UseZGC",
                "-XX:+ZGenerational",
                "-XX:ConcGCThreads=6",
                "-XX:ParallelGCThreads=8",
                "-Xlog:gc:file=" + str(out / (name + ".gc")),
                "-cp",
                cp[version],
                *["-D" + k + "=" + str(v) for k, v in settings.items()],
                PACKAGE + main,
                *rest,
            ],
            out / (name + ".log"),
        )

    if args.compare_routing:
        import report

        image = args.degraded_image.resolve()
        print(
            "cache image sha256",
            hashlib.sha256(image.read_bytes()).hexdigest(),
            flush=True,
        )
        for version in ("base", "checkout"):
            launch(version, version, dict(restore=image, recovery=args.seconds))
        from feedback import summarize

        for version in ("base", "checkout"):
            summarize(
                out / (version + "-feedback.bin"), args.seconds - 120, args.seconds
            )
        report.compare_exit(
            out, args.seconds, args.max_lag_ms, versions=("base", "checkout")
        )
        return

    if args.exit_control:
        import report

        src = out / "exit-control-src"
        classes = out / "exit-control"
        src.mkdir()
        classes.mkdir()
        names = ["EngineQueueAdmission.java", "PrefillEndpoint.java"]
        endpoints = MODULE / "flexlb-sync/src/main/java/org/flexlb/balance/endpoint"
        for name in names:
            shutil.copyfile(endpoints / name, src / name)
        subprocess.run(
            ["git", "apply", HERE / "idle-exit.patch"],
            cwd=src,
            check=True,
            env=os.environ | {"GIT_CEILING_DIRECTORIES": str(src.parent)},
        )
        run(
            [
                jdk / "javac",
                "-proc:none",
                "-cp",
                cp["checkout"],
                "-d",
                classes,
                *[src / name for name in names],
            ]
        )
        cp["idle-exit"] = os.pathsep.join((str(classes), cp["checkout"]))
        if args.build_only:
            return
        image = args.degraded_image
        if image is None:
            image = out / "degraded.image"
            launch("base", "capture-degraded", dict(capture=image))
        image = image.resolve()
        print(
            "cache image sha256",
            hashlib.sha256(image.read_bytes()).hexdigest(),
            flush=True,
        )
        for variant in ("checkout", "idle-exit"):
            launch(variant, variant, dict(restore=image, recovery=args.seconds))
        report.compare_exit(out, args.seconds, args.max_lag_ms)
        return

    if args.arrivals:
        trace = args.arrivals.resolve()
        extra = dict(
            arrivals=trace,
            families=1500,
            copies=24,
            privateBlocks=14,
            templateDepth=1200,
            rotateCopies=args.rotate_copies,
        )
        if args.geometry:
            geometry = out / "geometry.bin"
            write_geometry(args.geometry, geometry)
            extra["geometry"] = geometry
        versions = list(cp) if args.trace_version == "both" else [args.trace_version]
        for version in versions:
            launch(version, version, extra)
        summarize_trace(out, versions)
        return

    import report

    for history, image in (
        ("degraded", args.degraded_image),
        ("paced", args.paced_image),
    ):
        if image is None:
            image = out / (history + ".image")
            launch(
                "base",
                "capture-" + history,
                dict(
                    capture=image,
                    pacedHistory=str(history == "paced").lower(),
                    moreWork=str(history != "paced").lower(),
                ),
            )
        image = image.resolve()
        print(
            "cache image",
            history,
            hashlib.sha256(image.read_bytes()).hexdigest(),
            flush=True,
        )
        for version in cp:
            launch(
                version,
                version + "-" + history,
                dict(restore=image, recovery=args.seconds),
            )
        if history == "degraded":
            for policy in ("fifo", "benefit", "benefit-overlap", "fifo-cap9"):
                launch(
                    "base",
                    "service-" + policy,
                    main="ServiceControl",
                    rest=(image, policy, "240"),
                )
    report.check(out, args.seconds, args.max_lag_ms)


def summarize_trace(out, versions):
    digests = []
    for version in versions:
        log = out / (version + ".log")
        rows, digest, elapsed, good = {}, None, {}, {}
        for line in log.open():
            if "NoClassDefFoundError" in line or "ClassNotFoundException" in line:
                raise ValueError(f"{version} replay class loading failed")
            if line.startswith("ROW,"):
                r = list(map(int, line.strip().split(",")[1:]))
                rows[r[0]] = r
            elif line.startswith("INPUT_DIGEST "):
                digest = line.split()[1:]
            elif line.startswith("WALL "):
                _, t, ms, _, _ = line.split()
                elapsed[int(t)] = int(ms)
            elif line.startswith("GOODPUT "):
                _, t, n = line.split()
                good[int(t)] = int(n)
        if digest is None:
            raise ValueError("unfinished trace: " + version)
        last = rows[max(rows)]
        if last[1] != int(digest[0]) or last[1] != sum(last[3:6]) or last[6] or last[7]:
            raise ValueError(f"{version} trace did not settle")
        digests.append(tuple(digest))
        with (out / (version + ".csv")).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "second",
                    "wall_ms",
                    "arrived",
                    "completed",
                    "rejected",
                    "engine_errors",
                    "master_pending",
                    "worker_waiting",
                    "fresh_tokens",
                    "prefill_input_tokens",
                    "success_input_tokens",
                ]
            )
            for sec, r in rows.items():
                writer.writerow(
                    [
                        sec,
                        elapsed[sec],
                        r[1],
                        r[3],
                        r[4],
                        r[5],
                        r[6],
                        r[7],
                        r[8],
                        r[9],
                        good[sec],
                    ]
                )
        print(
            version,
            "received/completed/rejected/engine_errors",
            [r[i] for i in (1, 3, 4, 5)],
        )
    if len(set(digests)) != 1:
        raise ValueError("the empirical inputs differ between variants")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--ref", default=BASE_REF)
    p.add_argument("--out", type=Path, help="new result directory")
    p.add_argument(
        "--maven-arg",
        action="append",
        default=[],
        help="repeat for Maven settings or offline options",
    )
    p.add_argument("--heap", default="64g")
    p.add_argument("--seed", type=int, default=621)
    p.add_argument(
        "--seconds", type=int, default=600, help="seconds after restoring the state"
    )
    p.add_argument("--max-lag-ms", type=int, default=2000)
    p.add_argument(
        "--arrivals", type=Path, help="gzip CSV of arrival_ms,priority,seq_tokens"
    )
    p.add_argument("--geometry", type=Path, help="gzip JSON anonymous block dictionary")
    p.add_argument(
        "--trace-version",
        choices=("base", "checkout", "both"),
        default="both",
        help="routing version(s) for empirical arrivals",
    )
    p.add_argument("--rotate-copies", type=int, default=0)
    p.add_argument(
        "--degraded-image", type=Path, help="optional previously captured LRU state"
    )
    p.add_argument("--paced-image", type=Path)
    p.add_argument("--build-only", action="store_true")
    p.add_argument(
        "--compare-routing",
        action="store_true",
        help="compare e44 and checkout from the same --degraded-image at 240/s",
    )
    p.add_argument(
        "--exit-control",
        action="store_true",
        help="compare waiting-only vs waiting-and-running exit from the same degraded state",
    )
    args = p.parse_args()
    if (
        (args.exit_control and args.arrivals)
        or args.seconds < 240
        or args.max_lag_ms <= 0
        or (args.geometry and not args.arrivals)
    ):
        p.error(
            "use >=240 seconds, a positive clock tolerance and supply --arrivals with --geometry"
        )
    if args.compare_routing and (
        not args.degraded_image or args.arrivals or args.exit_control
    ):
        p.error(
            "--compare-routing requires --degraded-image and a synthetic recovery tail"
        )
    replay(args)
