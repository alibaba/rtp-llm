"""Validate settled inputs and measured delivery times before comparing throughput."""

from pathlib import Path


def parse(path):
    rows, wall, charged, delivered, lag = {}, {}, {}, {}, {}
    digest = prefix = None
    with path.open() as stream:
        for line in stream:
            if any(
                error in line
                for error in (
                    "NoClassDefFoundError",
                    "ClassNotFoundException",
                    "Exception in thread",
                )
            ):
                raise ValueError(
                    f"{path.name} contains an uncaught or class-loading error"
                )
            if line.startswith("ROW,"):
                row = list(map(int, line.rstrip().split(",")[1:]))
                rows[row[0]] = row
            elif line.startswith("WALL "):
                _, second, ms, _, delay = line.split()
                wall[int(second)] = int(ms)
                lag[int(second)] = int(delay)
            elif line.startswith("OFFERED_TOKENS "):
                _, second, value = line.split()
                charged[int(second)] = int(value)
            elif line.startswith("GOODPUT "):
                _, second, value = line.split()
                delivered[int(second)] = int(value)
            elif line.startswith("TAIL_DIGEST "):
                digest = line.split()[1]
            elif line.startswith("PREFIX_DIGEST_57600 "):
                prefix = line.split()[1]
    last = rows[max(rows)]
    if not digest or last[1] != sum(last[3:6]) or last[6] or last[7]:
        raise ValueError(f"{path.name} did not settle")
    return digest, prefix, rows, wall, charged, delivered, lag


def check(out: Path, seconds: int, max_lag_ms: int):
    policies = ("fifo", "benefit", "benefit-overlap", "fifo-cap9")
    references = []
    for policy in policies:
        line = next(
            line
            for line in (out / f"service-{policy}.log").read_text().splitlines()
            if line.startswith("TAIL ")
        )
        print(policy, line)
        references.append(dict(item.split("=", 1) for item in line.split()[1:]))
    if len({r["digest"] for r in references}) != 1:
        raise ValueError("service controls used different inputs")
    witness = references[2]
    if (
        float(witness["complete"]) < 0.99 * float(witness["offered"])
        or float(witness["goodTokensM"]) < 0.99 * float(witness["offeredTokensM"])
        or float(witness["reject"]) > 2.4
        or int(witness["late"])
    ):
        raise ValueError("the service control did not demonstrate recovery")
    digests = set()
    failed = []
    end = 155 + seconds
    for history in ("degraded", "paced"):
        for version in ("base", "checkout"):
            name = f"{version}-{history}"
            digest, prefix, rows, wall, offered, good, lag = parse(
                out / (name + ".log")
            )
            digests.add(digest)
            if prefix != witness["digest"] or max(lag.values()) > max_lag_ms:
                raise ValueError(f"{name} stream or wall clock mismatch")
            for start, stop in sorted({(275, 395), (end - 120, end), (end - 40, end)}):
                a, b = rows[start], rows[stop]
                elapsed = (wall[stop] - wall[start]) / 1000
                rate = lambda i: (b[i] - a[i]) / elapsed
                useful = (good[stop] - good[start]) / elapsed
                demand = (offered[stop] - offered[start]) / elapsed
                print(
                    f"{name} {start}..{stop} offered={rate(1):.2f} complete={rate(3):.2f} "
                    f"goodTokensM={useful/1e6:.3f} offeredTokensM={demand/1e6:.3f} "
                    f"meanCompletedLen={useful/max(1.,rate(3)):.0f} freshM={rate(8)/1e6:.3f} "
                    f"hit={100*rate(10)/max(1.,rate(9)):.2f} reject={rate(4):.2f} "
                    f"engineErrors={rate(5):.2f} master={a[6]}->{b[6]} worker={a[7]}->{b[7]}"
                )
                if abs(rate(1) - 240) > 2:
                    raise ValueError(f"{name} offered rate diverged")
                if (
                    version == "checkout"
                    and stop == end
                    and (
                        rate(3) < 0.99 * rate(1)
                        or useful < 0.99 * demand
                        or rate(4) > 0.01 * rate(1)
                        or rate(5) > 0
                        or b[6] > min(a[6] + 30, 240)
                    )
                ):
                    failed.append((name, start, stop))
    if len(digests) != 1:
        raise ValueError("different tail inputs in the four routing runs")
    if failed:
        raise SystemExit(f"240/s recovery failed in {failed}")


def compare_exit(
    out: Path, seconds: int, max_lag_ms: int, versions=("checkout", "idle-exit")
):
    digests = set()
    end = 155 + seconds
    misses = []
    for name in versions:
        digest, _, rows, wall, offered, good, lag = parse(out / (name + ".log"))
        digests.add(digest)
        if max(lag.values()) > max_lag_ms:
            raise ValueError(f"{name} arrival clock exceeded tolerance")
        for start in (end - 120, end - 40):
            a, b = rows[start], rows[end]
            t = (wall[end] - wall[start]) / 1000
            if abs((b[1] - a[1]) / t - 240) > 2:
                raise ValueError(f"{name} offered rate diverged")
            print(
                f"{name} {start}..{end} offered={(b[1]-a[1])/t:.2f} completed={(b[3]-a[3])/t:.2f} "
                f"completedInputM={(good[end]-good[start])/t/1e6:.3f} rejected={(b[4]-a[4])/t:.2f} "
                f"freshM={(b[8]-a[8])/t/1e6:.3f} master={a[6]}->{b[6]} worker={a[7]}->{b[7]}"
            )
            if name == "checkout":
                completions = b[3] - a[3]
                incoming = b[1] - a[1]
                if (
                    completions < 0.99 * incoming
                    or b[4] - a[4] > 0.01 * incoming
                    or b[5] != a[5]
                    or b[6] > min(a[6] + 30, 240)
                ):
                    misses.append((start, end))
                elif start not in offered or end not in offered:
                    raise ValueError(
                        "missing input token counters for a potential PASS"
                    )
                elif good[end] - good[start] < 0.99 * (offered[end] - offered[start]):
                    misses.append((start, end))
    if len(digests) != 1:
        raise ValueError("routing runs received different inputs")
    if misses:
        raise SystemExit(f"FAIL: checkout did not recover at 240/s in {misses}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    p = ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument("--max-lag-ms", type=int, default=2000)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--exit-control", action="store_true")
    mode.add_argument("--compare-routing", action="store_true")
    args = p.parse_args()
    if args.compare_routing:
        compare_exit(
            args.directory, args.seconds, args.max_lag_ms, ("base", "checkout")
        )
    elif args.exit_control:
        compare_exit(args.directory, args.seconds, args.max_lag_ms)
    else:
        check(args.directory, args.seconds, args.max_lag_ms)
