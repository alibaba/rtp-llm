"""Read K3 pool allocation from the native engine log."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re


LOG_HEADER = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
RANK = re.compile(r"\[RANK (\d+)\]")
FIELDS = re.compile(r"\b(\w+)=([^\s]+)")
POOL_TYPES = {0: "LINEAR", 1: "FULL", 2: "SWA"}


def parse_config(record):
    """Only the merged config's outer groups describe the allocated device pools."""
    config, specs, spec = {}, {}, None
    for line in record.splitlines():
        match = re.fullmatch(r"  cache_specs\[(\d+)\] \{", line)
        if match:
            spec = specs.setdefault(int(match[1]), {})
        elif line == "  }":
            spec = None
        elif (
            line.startswith("    ")
            and not line.startswith("     ")
            and spec is not None
        ):
            key, sep, value = line.strip().partition("=")
            if sep:
                spec[key] = value
        elif line.startswith("  ") and not line.startswith("   "):
            key, sep, value = line.strip().partition("=")
            if sep:
                config[key] = value
    config["specs"] = specs
    reserve = re.search(r"^reserve_block_ratio: (\d+)$", record, re.MULTILINE)
    if reserve is not None:
        config["reserve_block_ratio"] = int(reserve[1])
    return config


class CacheEvidence:
    def __init__(self):
        self.configs = {}
        self.backings = {}
        self.reserves = {}
        self.reserve_ratios = {}
        self.reserve_steps = {}
        self._config_record = []
        self._config_rank = None

    def finish_record(self):
        if self._config_record:
            self.configs[self._config_rank] = parse_config("".join(self._config_record))
            self._config_record = []

    def feed(self, line, location):
        header = LOG_HEADER.search(line)
        if header:
            if self._config_record and header.start():
                self._config_record.append(line[:header.start()])
            self.finish_record()
            line = line[header.start():]
        rank_match = RANK.search(line)
        rank = int(rank_match[1]) if rank_match else None
        if "init connector coordinator, cache config: [CacheConfig{" in line:
            if rank is None:
                raise ValueError(f"merged config has no rank: {location}")
            self._config_rank = rank
            self._config_record = [line]
            return
        if self._config_record:
            self._config_record.append(line)
            return
        if not header:
            return
        fields = dict(FIELDS.findall(line))
        if "BlockPool backing selected:" in line:
            key = (rank, fields["pool_name"])
            backing = dict(fields, rank=rank, source=location)
            previous = self.backings.get(key)
            if previous and any(
                previous[k] != backing[k]
                for k in (
                    "allocation_type",
                    "actual_backing",
                    "total_size",
                    "block_num",
                )
            ):
                raise ValueError(
                    f"pool was reallocated with a different configuration: {key}"
                )
            self.backings[key] = backing
        elif "KVCacheAllocator set reserve blocks:" in line:
            self.reserves[rank] = int(fields["reserve_blocks"])
            self.reserve_ratios[rank] = int(fields["ratio"].rstrip("%"))
        elif "normal engine speculative reserve_step is " in line:
            self.reserve_steps[rank] = int(re.search(r"reserve_step is (\d+)", line)[1])

    def inventory(self, role, tp_size, cp_size):
        if set(self.configs) != set(range(tp_size)):
            raise ValueError(
                f"missing merged configs: ranks={sorted(self.configs)}, expected={tp_size}"
            )
        rows = []
        for rank, config in sorted(self.configs.items()):
            # Multiple ranks can interleave the long config dump before its
            # trailing KVCacheConfig. The allocator's single-line result is
            # authoritative and carries both the ratio and actual reserve.
            reserve_ratio = self.reserve_ratios.get(rank, config.get("reserve_block_ratio"))
            if reserve_ratio is None:
                raise ValueError(f"missing reserve budget evidence for rank {rank}")
            counts = [
                int(value) for value in next(csv.reader([config["group_block_nums"]]))
            ]
            kinds = json.loads(config["group_types"])
            layers = json.loads(config["global_layer_ids"])
            if not counts or len(counts) != len(kinds) or len(counts) != len(layers):
                raise ValueError(f"inconsistent group arrays for rank {rank}")
            for group, (count, kind, layer_ids) in enumerate(
                zip(counts, kinds, layers)
            ):
                spec = config["specs"][group]
                backing = self.backings.get((rank, f"group_{group}"))
                if count <= 0 or backing is None or int(backing["block_num"]) != count:
                    raise ValueError(
                        f"missing/mismatched allocated pool: rank={rank}, group={group}"
                    )
                if backing["allocation_type"] != "DEVICE" or backing["is_cuda"] != "1":
                    raise ValueError(f"expected GPU pool: rank={rank}, group={group}")
                if int(spec["layer_num"]) != len(layer_ids):
                    raise ValueError(
                        f"layer mapping mismatch: rank={rank}, group={group}"
                    )
                rows.append(
                    dict(
                        role=role,
                        rank=rank,
                        group=group,
                        type=POOL_TYPES[kind],
                        configured_blocks=count,
                        usable_blocks=count - 1,
                        allocated_bytes=int(backing["total_size"]),
                        group_block_bytes=int(backing["total_size"]) // count,
                        spec_type=spec.get("type"), dtype=spec.get("dtype"),
                        ssm_state_dtype=spec.get("ssm_state_dtype"), conv_state_dtype=spec.get("conv_state_dtype"),
                        reuse_cache=(config.get("reuse_cache") in ("1", "true") if "reuse_cache" in config else None),
                        speculative_reserve_step=self.reserve_steps.get(rank),
                        cache_region=(json.loads(config["group_region_names"])[group] if "group_region_names" in config else None),
                        physical_page_tokens=int(config["seq_size_per_block"]),
                        kernel_page_tokens=int(config["kernel_seq_size_per_block"]),
                        cp_size=cp_size,
                        virtual_page_tokens=int(config["seq_size_per_block"]) * cp_size,
                        spec_seq_size_per_block=int(spec["seq_size_per_block"]),
                        logical_block_tokens=int(spec["seq_size_per_block"])
                        * (cp_size if kind == 1 else 1),
                        layer_ids=layer_ids,
                        linear_step=int(config["linear_step"]),
                        reserve_ratio_percent=reserve_ratio,
                        allocator_reserve_blocks=(
                            0
                            if reserve_ratio == 0
                            else self.reserves[rank]
                        ),
                        source=backing["source"],
                    )
                )
        reference = [
            {k: v for k, v in row.items() if k not in ("rank", "source", "speculative_reserve_step")}
            for row in rows
            if row["rank"] == 0
        ]
        for rank in range(1, tp_size):
            other = [
                {k: v for k, v in row.items() if k not in ("rank", "source", "speculative_reserve_step")}
                for row in rows
                if row["rank"] == rank
            ]
            if other != reference:
                raise ValueError(f"rank {rank} pool allocation differs from rank 0")
        return rows

def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-log", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--cp-size", type=int, required=True)
    parser.add_argument("--phase", choices=("startup", "final"), required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence = CacheEvidence()
    with args.engine_log.open() as stream:
        for number, line in enumerate(stream, 1):
            evidence.feed(line, f"{args.engine_log}:{number}")
    evidence.finish_record()
    inventory = evidence.inventory(args.role, args.tp_size, args.cp_size)
    payload = dict(
        role=args.role,
        inventory=inventory,
        backings=list(evidence.backings.values()),
    )
    destination = args.output_dir / f"{args.phase}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(args.output_dir / "pool_inventory.csv", inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
