import csv
import hashlib
import json
import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

BUCKET_WIDTH = 1024

KNOWN_DATASETS: Dict[str, Tuple[str, str]] = {
    "sharegpt": (
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "ShareGPT_V3_unfiltered_cleaned_split.json",
    ),
}

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "rtp_llm", "perf")


class DatasetLoader:
    """Dataset acquisition, format detection, tokenization, and histogram building."""

    def __init__(
        self,
        tokenizer_path: str,
        dataset_name: str = "",
        dataset_path: str = "",
    ):
        if dataset_name:
            self._path = self._download(dataset_name)
        elif dataset_path:
            self._path = dataset_path
        else:
            raise ValueError("Either dataset_name or dataset_path must be provided")
        self._tokenizer_path = tokenizer_path

    def load_histogram(self) -> Tuple[List[int], List[int]]:
        """Load JSON -> detect format -> tokenize -> return (uppers, counts).

        Results are cached under ~/.cache/rtp_llm/perf/ keyed by
        (dataset path, file size, mtime, tokenizer_path).
        """
        cached = self._load_from_cache()
        if cached is not None:
            return cached

        from rtp_llm.test.perf_test.test_util import _load_tokenizer

        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)

        prompts = self._extract_prompts(data)
        logging.info(f"Extracted {len(prompts)} prompts from {self._path}")

        tokenizer = _load_tokenizer(self._tokenizer_path)

        bucket_counts: Counter = Counter()
        total = len(prompts)
        for idx, prompt in enumerate(prompts):
            if idx % 1000 == 0 and idx > 0:
                logging.info(f"Tokenizing: {idx}/{total} prompts processed")
            prompt_len = len(tokenizer.encode(prompt))
            if prompt_len < 4:
                continue
            bucket_upper = ((prompt_len - 1) // BUCKET_WIDTH + 1) * BUCKET_WIDTH
            bucket_counts[bucket_upper] += 1

        uppers = sorted(bucket_counts.keys())
        counts = [bucket_counts[u] for u in uppers]
        logging.info(
            f"Built histogram: {len(uppers)} buckets, {sum(counts)} total samples"
        )
        self._save_to_cache(uppers, counts)
        return uppers, counts

    def _cache_key(self) -> str:
        abs_path = os.path.abspath(self._path)
        try:
            stat = os.stat(abs_path)
            key = f"{abs_path}:{stat.st_size}:{stat.st_mtime_ns}:{self._tokenizer_path}"
        except OSError:
            key = f"{abs_path}:{self._tokenizer_path}"
        return hashlib.md5(key.encode()).hexdigest()

    def _cache_path(self) -> str:
        return os.path.join(CACHE_DIR, f"histogram_{self._cache_key()}.json")

    def _load_from_cache(self) -> Optional[Tuple[List[int], List[int]]]:
        path = self._cache_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            uppers, counts = data["uppers"], data["counts"]
            logging.info(
                f"Loaded cached histogram from {path} "
                f"({len(uppers)} buckets, {sum(counts)} samples)"
            )
            return uppers, counts
        except Exception as e:
            logging.warning(f"Failed to load histogram cache {path}: {e}")
            return None

    def _save_to_cache(self, uppers: List[int], counts: List[int]) -> None:
        path = self._cache_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    {
                        "dataset_path": os.path.abspath(self._path),
                        "tokenizer_path": self._tokenizer_path,
                        "uppers": uppers,
                        "counts": counts,
                    },
                    f,
                )
            logging.info(f"Cached histogram to {path}")
        except Exception as e:
            logging.warning(f"Failed to cache histogram: {e}")

    @staticmethod
    def save_histogram_csv(
        uppers: List[int], counts: List[int], output_path: str
    ) -> None:
        """Save histogram to distribution.csv for reproducibility."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["bucket_id", "length_range", "cnt"])
            for upper, count in zip(uppers, counts):
                bid = (upper // BUCKET_WIDTH) - 1
                lower = upper - BUCKET_WIDTH
                writer.writerow([str(bid), f"{lower}-{upper}", str(count)])
        logging.info(f"Saved histogram CSV to {output_path}")

    @staticmethod
    def load_histogram_csv(csv_path: str) -> Tuple[List[int], List[int]]:
        r"""Read distribution.csv -> (bucket_upper_bounds, counts).
        Bucket boundaries are multiples of BUCKET_WIDTH. Skips \N rows."""
        uppers: List[int] = []
        counts: List[int] = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                bid = row["bucket_id"].strip('"')
                if bid == "\\N" or bid == "":
                    continue
                _, hi = row["length_range"].strip('"').split("-")
                uppers.append(int(hi))
                counts.append(int(row["cnt"].strip('"')))
        return uppers, counts

    @staticmethod
    def _download(dataset_name: str, timeout: int = 300) -> str:
        """Auto-download dataset JSON, return local path.

        Try order: ModelScope -> HuggingFace（实现见 ``rtp_llm.test.perf_test.hub_download``）；
        下载前用 curl 探测连通性，下载过程用 SIGALRM 限制阻塞时间。
        """
        if dataset_name in KNOWN_DATASETS:
            repo_id, filename = KNOWN_DATASETS[dataset_name]
        else:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Known: {list(KNOWN_DATASETS.keys())}. "
                f"Or provide --dataset_path directly."
            )

        from rtp_llm.test.perf_test.hub_download import download_dataset_repo_file

        return download_dataset_repo_file(
            repo_id,
            filename,
            repo_type="dataset",
            label=dataset_name,
            timeout=timeout,
        )

    @staticmethod
    def _extract_prompts(data: List[Any]) -> List[str]:
        """Extract prompt strings from dataset JSON (following vLLM conventions).

        Supported formats:
        - Conversation (ShareGPT): [{"conversations": [{"value": "..."}, ...]}]
        - Prompt-based: [{"prompt": "..."}]  or  [{"text": "..."}]
        """
        if not data or not isinstance(data[0], dict):
            raise ValueError("Dataset must be a JSON array of objects")

        sample = data[0]

        if "conversations" in sample:
            filtered = [
                d for d in data if "conversations" in d and len(d["conversations"]) >= 2
            ]
            logging.info(
                f"Conversation format: {len(filtered)}/{len(data)} "
                f"entries with >=2 turns"
            )
            return [d["conversations"][0]["value"] for d in filtered]

        for key in ("prompt", "text"):
            if key in sample:
                prompts = [d[key] for d in data if key in d and d[key]]
                logging.info(f"Prompt format ('{key}' field): {len(prompts)} entries")
                return prompts

        raise ValueError(
            f"Unsupported format (keys: {list(sample.keys())}). "
            f"Expected 'conversations' (ShareGPT), 'prompt', or 'text'."
        )


def extract_arg(
    args_list: List[str], key: str, default: Optional[str] = None
) -> Optional[str]:
    """Extract --key value from a CLI args list (without removing it)."""
    flag = f"--{key}"
    prefix = f"--{key}="
    for i, arg in enumerate(args_list):
        if arg == flag and i + 1 < len(args_list):
            return args_list[i + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return default
