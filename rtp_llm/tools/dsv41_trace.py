"""Opt-in model-boundary snapshots for native/reference accuracy qualification."""

from contextlib import ExitStack, contextmanager


@contextmanager
def capture_model_boundaries(model):
    tensors = {}

    def before(name):
        def hook(module, inputs):
            tensors[name] = inputs[0].detach().clone()

        return hook

    def after(name):
        def hook(module, inputs, output):
            tensors[name] = output.detach().clone()

        return hook

    with ExitStack() as stack:
        for layer in model.layers:
            label = f"layer{layer.layer_id:02d}.ffn"
            stack.callback(
                layer.ffn.register_forward_pre_hook(before(label + ".input")).remove
            )
            stack.callback(
                layer.ffn.register_forward_hook(after(label + ".output")).remove
            )
        stack.callback(model.norm.register_forward_hook(after("final_norm")).remove)
        yield tensors
    # Defer device-to-host copies until the complete forward has returned.
    for name, value in tensors.items():
        tensors[name] = value.cpu()


def save_model_trace(path, tensors, metadata):
    import torch

    if any(value.device.type != "cpu" for value in tensors.values()):
        raise ValueError("Model trace tensors must be detached CPU snapshots")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save({"tensors": tensors, "metadata": metadata}, handle)
