import torch


def warmup(vocab_size: int) -> str:
    """Return a dependency/compiler failure; let CUDA execution failures propagate."""
    from triton.compiler.errors import CompilationError

    try:
        from rtp_llm.models_py.sampling import adapter

        vocab_size = max(vocab_size, 8193)
        logits = torch.zeros((1, 2, vocab_size), device="cuda", dtype=torch.float32)
        seeds = torch.zeros(1, device="cuda", dtype=torch.int64)
        positions = torch.zeros_like(seeds)
        draft_ids = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
        dense = torch.full((1, 1, vocab_size), 1.0 / vocab_size, device="cuda")
        for draft in (None, dense):
            for random in (False, True):
                adapter.rejection_sample_logits(
                    logits,
                    draft,
                    draft_ids,
                    seeds,
                    positions,
                    torch.full((1,), random, device="cuda", dtype=torch.bool),
                )
        torch.cuda.current_stream().synchronize()
    except (ImportError, CompilationError) as error:
        return f"{type(error).__name__}: {error}"
    return ""
