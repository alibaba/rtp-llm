# based on flashinfer 0.4.1 https://github.com/flashinfer-ai/flashinfer/tree/a88349f9f43df74d31d1d52ad5aa20c28824a790
"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

from bind import (
    get_seed_and_offset,
    top_p_renorm_probs,
    top_k_renorm_probs,
    top_p_sampling_from_probs,
    top_k_sampling_from_probs,
    top_k_top_p_sampling_from_probs,
)
import json
import os
from pathlib import Path


def normal_distribution(std):
    def normal_noise(shape, device):
        return torch.randn(shape, device=device) * std

    normal_noise.__name__ = f"normal_distribution(std={std})"
    return normal_noise


def gumbel_distribution(beta):
    def gumbel_noise(shape, device):
        U = torch.rand(shape, device=device)
        eps = 1e-20
        return torch.log(-torch.log(U + eps) + eps) / beta

    gumbel_noise.__name__ = f"gumbel_distribution(beta={beta})"
    return gumbel_noise

@pytest.mark.parametrize("batch_size", [1, 2, 10])
def test_get_seed_and_offset(batch_size):
    def get_seed_and_offset_ref(increment, generator = None):
        if generator is None:
            generator = torch.Generator("cuda:0")
        state = generator.get_state()
        seed, offset = state.view(torch.int64)
        generator.set_state(
            torch.tensor([seed, offset + (increment + 3) // 4 * 4], dtype=torch.int64).view(torch.uint8)
        )
        return int(seed), int(offset)

    gen1 = torch.Generator("cuda:0")
    gen2 = gen1.clone_state()
    increment = 32 * batch_size

    seed_ref, offset_ref = get_seed_and_offset_ref(increment, gen1)
    seed, offset = get_seed_and_offset(increment, gen2)
    assert seed == seed_ref, f"seed={seed}, seed_ref={seed_ref}"
    assert offset == offset_ref, f"offset={offset}, offset_ref={offset_ref}"

@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_sampling_freq(vocab_size, distribution, p):
    # use torch profiler to check the performance of the code
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)
    sorted_prob, indices = torch.sort(probs, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(1, vocab_size, dtype=torch.int32, device=logits.device)
    mask.scatter_add_(1, indices, (cdf > (1 - p)).int())

    renorm_probs = torch.zeros_like(probs)
    top_p_renorm_probs(probs, renorm_probs, None, p)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)
    num_trials = 3000000 # 5000000 core
    samples = torch.empty(num_trials, dtype=torch.int32, device="cuda:0")
    top_p_sampling_from_probs(
        probs,
        samples,
        torch.zeros(num_trials, dtype=torch.int32, device=logits.device),
        None,
        p,
        False,
        torch.zeros(num_trials, dtype=torch.uint64, device="cuda:0"),
        torch.zeros(num_trials, dtype=torch.uint64, device="cuda:0"),
    )
    counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    freq = counter.float() / num_trials
    assert torch.all(mask[torch.arange(1), samples] == 1)
    similarity = torch.cosine_similarity(freq, renorm_probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [
        normal_distribution(1),
        normal_distribution(5),
        gumbel_distribution(0.1),
    ],
)
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling_freq(vocab_size, distribution, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = torch.softmax(logits, dim=-1)
    sorted_prob, _ = torch.sort(probs, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (probs >= pivot.unsqueeze(-1)).int()

    renorm_probs = torch.zeros_like(probs)
    top_k_renorm_probs(probs, renorm_probs, None, k)
    counter = torch.zeros(vocab_size, dtype=torch.int32, device=logits.device)
    num_trials = 3000000 # 5000000 core
    samples = torch.empty(num_trials, dtype=torch.int32, device="cuda:0")
    top_k_sampling_from_probs(
        probs,
        samples,
        torch.zeros(num_trials, dtype=torch.int32, device=logits.device),
        None,
        k,
        False,
        torch.zeros(num_trials, dtype=torch.uint64, device="cuda:0"),
        torch.zeros(num_trials, dtype=torch.uint64, device="cuda:0"),
    )
    counter.scatter_add_(0, samples.long(), torch.ones_like(samples))
    freq = counter.float() / num_trials
    assert torch.all(mask[torch.arange(1), samples] == 1)
    similarity = torch.cosine_similarity(freq, renorm_probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_sampling(batch_size, vocab_size, p):
    torch.manual_seed(42)
    info = dict()
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    # single_prob = torch.rand(vocab_size, device="cuda:0")
    # pre_norm_prob = single_prob.unsqueeze(0).expand(batch_size, -1).contiguous()
    # pre_norm_prob = torch.ones_like(pre_norm_prob)
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    realdata = os.environ.get("REALDATA")
    if realdata:
        realdata = Path(realdata)
        assert realdata.is_file()
        normalized_prob = torch.load(realdata, weights_only=True).to("cuda:0")
    
    info["prob"] = normalized_prob.cpu().numpy().tolist()
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    info["sorted_prob"] = sorted_prob.cpu().numpy().tolist()
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    info["mask"] = mask.cpu().numpy().tolist()
    file = os.environ.get("DEBUG_SAMPLE_INFO")
    if file:
      file = Path(file)
      with file.open("w") as f:
          json.dump(info, f, ensure_ascii=False, indent=4)
    num_trials = 1000
    info["out"] = []
    for _ in range(num_trials):
        samples = torch.empty(batch_size, dtype=torch.int32, device="cuda:0")
        top_p_sampling_from_probs(
            normalized_prob,
            samples,
            None,
            None,
            p,
            False,
            torch.zeros(batch_size, dtype=torch.uint64, device="cuda:0"),
            torch.zeros(batch_size, dtype=torch.uint64, device="cuda:0"),
        )
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1)
        info["out"].append(samples.cpu().numpy().tolist())
    if file:
        with file.open("w") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()

    num_trials = 1000
    for _ in range(num_trials):
        samples = torch.empty(batch_size, dtype=torch.int32, device="cuda:0")
        top_k_sampling_from_probs(
            normalized_prob,
            samples,
            None,
            None,
            k,
            False,
            torch.zeros(batch_size, dtype=torch.uint64, device="cuda:0"),
            torch.zeros(batch_size, dtype=torch.uint64, device="cuda:0"),
        )
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling_with_variable_k(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    k = torch.randint(1, k + 1, (batch_size,), device="cuda:0", dtype=torch.int32)
    pivot = sorted_prob[torch.arange(batch_size), k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()

    num_trials = 1000
    for _ in range(num_trials):
        samples = torch.empty(batch_size, dtype=torch.int32, device="cuda:0")
        top_k_sampling_from_probs(
            normalized_prob,
            samples,
            None,
            k,
            0,
            False,
            torch.zeros(batch_size, dtype=torch.uint64, device="cuda:0"),
            torch.zeros(batch_size, dtype=torch.uint64, device="cuda:0"),
        )
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_joint_sampling_from_probs(batch_size, vocab_size, p):
    torch.manual_seed(42)
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    # top-p mask
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask_top_p = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask_top_p.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    # top-k mask
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask_top_k = (normalized_prob >= pivot.unsqueeze(-1)).int()
    # overall mask
    mask = torch.minimum(mask_top_p, mask_top_k)
    top_p_tensor = torch.full((batch_size,), p, device="cuda:0")
    top_k_tensor = torch.full((batch_size,), k, device="cuda:0", dtype=torch.int32)

    num_trials = 1000
    for _ in range(num_trials):
        samples = torch.empty(batch_size, dtype=torch.int32, device="cuda:0")
        top_k_top_p_sampling_from_probs(
            normalized_prob,
            samples,
            None,
            top_k_tensor,
            0,
            top_p_tensor,
            0,
            False,
            torch.zeros(batch_size, dtype=torch.uint64, device="cuda:0"),
            torch.zeros(batch_size, dtype=torch.uint64, device="cuda:0"),
        )
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 1.0])
def test_top_p_renorm_probs(batch_size, vocab_size, p):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf >= (1 - p)).int())
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = torch.zeros_like(normalized_prob)
    top_p_renorm_probs(normalized_prob, renorm_prob, None, p)
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_renorm_probs(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = torch.zeros_like(normalized_prob)
    top_k_renorm_probs(normalized_prob, renorm_prob, None, k)
    for i in range(batch_size):
        torch.testing.assert_close(
            renorm_prob_ground_truth[i],
            renorm_prob[i],
            rtol=1e-3,
            atol=1e-3,
        )


if __name__ == "__main__":
    exit(pytest.main([__file__, "-s", "-v"]))
