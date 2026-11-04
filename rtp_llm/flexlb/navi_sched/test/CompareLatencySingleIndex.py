#!/usr/bin/env python3
"""Test whether a learned scalar workload plus a flexible curve can explain latency."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from AnalyzeLatencyPhaseLeastSquares import PARAMETER_SCALES, load_host, metrics
from CompareLatencyAugmentedSquareShape import inner_input
from PlotLatencyHighPressureU import operating_masks


FEATURE_COLUMNS = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13)


class SingleIndexModel(nn.Module):
    def __init__(self, feature_count, index_dim=1):
        super().__init__()
        self.index = nn.Linear(feature_count, index_dim)
        self.curve = nn.Sequential(
            nn.Linear(index_dim, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(),
            nn.Linear(32, 1))
        self.bypass = nn.Linear(2, 1)

    def forward(self, features, bypass):
        return self.curve(self.index(features)) + self.bypass(bypass)


def prepare(inputs, train):
    features = inputs[:, FEATURE_COLUMNS]
    bypass = np.column_stack((inputs[:, 2] + inputs[:, 3] + inputs[:, 8] + inputs[:, 9],
                              inputs[:, 12]))
    feature_mean = features[train].mean(axis=0)
    feature_scale = features[train].std(axis=0)
    feature_scale[feature_scale < 1e-9] = 1.0
    bypass_mean = bypass[train].mean(axis=0)
    bypass_scale = bypass[train].std(axis=0)
    bypass_scale[bypass_scale < 1e-9] = 1.0
    return ((features - feature_mean) / feature_scale,
            (bypass - bypass_mean) / bypass_scale)


def fit_predict(inputs, target, train, epochs, seeds, index_dim=1):
    torch.set_num_threads(1)
    feature_array, bypass_array = prepare(inputs, train)
    features = torch.from_numpy(feature_array.astype(np.float32))
    bypass = torch.from_numpy(bypass_array.astype(np.float32))
    response = torch.from_numpy((target / 50000.0).astype(np.float32)).reshape(-1, 1)
    selected = torch.from_numpy(train)
    best_loss = float("inf")
    best_prediction = None
    best_index = None
    for seed in range(seeds):
        torch.manual_seed(17 + seed)
        model = SingleIndexModel(features.shape[1], index_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        saved_loss = float("inf")
        saved_prediction = None
        saved_index = None
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predicted = model(features[selected], bypass[selected])
            loss = torch.mean((predicted - response[selected]) ** 2)
            loss.backward()
            optimizer.step()
            if epoch % 25 == 24 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    current = float(torch.mean((model(features[selected], bypass[selected]) -
                                               response[selected]) ** 2))
                    if current < saved_loss:
                        saved_loss = current
                        saved_prediction = model(features, bypass).numpy().ravel() * 50000.0
                        saved_index = model.index(features).numpy()
        if saved_loss < best_loss:
            best_loss = saved_loss
            best_prediction = saved_prediction
            best_index = saved_index
    return np.maximum(best_prediction, 0.0), best_index, best_loss


def evaluate_host(args):
    path, epochs, seeds, index_dim, high_fit = args
    inputs, target, event_time, _, _, _, _ = load_host(path)
    high = inputs[:, 7] >= 33
    x = inputs[high]
    y = target[high]
    time = event_time[high]
    regions = None
    if high_fit is not None:
        parameters = np.asarray(high_fit["fit_parameters"])
        regions = operating_masks(inner_input(x, parameters), parameters[8] * PARAMETER_SCALES[8])
    if len(y) < 100:
        raise ValueError(f"too few high-pressure samples: {path}")
    random = np.random.default_rng(17)
    random_train = np.zeros(len(y), dtype=bool)
    random_train[random.permutation(len(y))[:len(y) * 4 // 5]] = True
    time_train = np.zeros(len(y), dtype=bool)
    time_train[np.argsort(time)[:len(y) * 4 // 5]] = True
    experiments = {}
    for split, train in (("all_fit", np.ones(len(y), dtype=bool)),
                         ("random_split", random_train), ("time_split", time_train)):
        prediction, index, train_mse = fit_predict(x, y, train, epochs, seeds, index_dim)
        held = ~train
        experiments[split] = {"train": metrics(y[train], prediction[train]),
                              "held": metrics(y[held], prediction[held]) if held.any() else None,
                              "train_mse_normalized": train_mse,
                              "index_percentiles": [np.percentile(
                                  index[:, dimension], [5, 50, 95]).tolist()
                                  for dimension in range(index_dim)]}
        if regions is not None:
            experiments[split]["operating_regions"] = {
                name: {"train": metrics(y[train & region], prediction[train & region]),
                       "held": metrics(y[held & region], prediction[held & region])}
                for name, region in regions.items()}
    architecture = "MI" if path.stem.startswith("mi-") or "MI308" in path.stem else "RTX"
    return {"host": path.stem, "architecture": architecture,
            "high_sample_count": len(y), "experiments": experiments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--hosts", nargs="*")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--index-dim", type=int, default=1)
    parser.add_argument("--high-fit-json", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.seeds, args.workers, args.index_dim) < 1:
        parser.error("epochs, seeds, workers, and index-dim must be positive")
    paths = sorted(args.input_dir.glob("*.jsonl"))
    if args.hosts:
        paths = [path for path in paths if path.stem in args.hosts]
    if not paths:
        parser.error("no matching input JSONL files")
    high_fits = ({host["host"]: host for host in
                  json.loads(args.high_fit_json.read_text())["hosts"]}
                 if args.high_fit_json else {})
    if args.high_fit_json and set(path.stem for path in paths) - set(high_fits):
        parser.error("high-fit JSON is missing selected hosts")
    work = [(path, args.epochs, args.seeds, args.index_dim,
             high_fits.get(path.stem)) for path in paths]
    if args.workers == 1:
        hosts = [evaluate_host(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            hosts = list(executor.map(evaluate_host, work))
    print(json.dumps({"input_dir": str(args.input_dir), "epochs": args.epochs,
                      "seeds": args.seeds, "index_dim": args.index_dim,
                      "hosts": hosts}, separators=(",", ":")))


if __name__ == "__main__":
    main()
