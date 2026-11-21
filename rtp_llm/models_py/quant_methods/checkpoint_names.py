"""Module-local quantized checkpoint suffixes, without internal weight names."""

_CANONICAL_PARAMETERS = (
    "weight",
    "bias",
    "weight_scale",
    "input_scale",
    "weight_scale_inv",
    "weight_scale_2",
    "weight_packed",
    "weight_shape",
)


class QuantizedCheckpointNames:
    def __init__(self, family, quant_config):
        source = getattr(quant_config, "source_config", None)
        method = getattr(source, "get_method", None)
        method = method() if callable(method) else None
        quant_type = getattr(quant_config, "quant_type", None)
        configured = {}
        if family == "fp8_per_tensor" and (
            method == "FP8_PER_TENSOR_COMPRESSED"
            or quant_type == "FP8_PER_TENSOR_COMPRESSED"
        ):
            configured = {
                "weight_scale": getattr(source, "_weight_s_suffix", None),
                "input_scale": getattr(source, "_act_s_suffix", None),
            }
            if configured["weight_scale"] is None:
                configured["weight_scale"] = ".weights_scaling_factor"
            if configured["input_scale"] is None:
                configured["input_scale"] = ".activation_scaling_factor"
        elif (
            family == "w4a8_compressed"
            and quant_type == "W4A8_INT4_PER_CHANNEL_COMPRESSED"
        ):
            configured = {
                "weight_packed": getattr(
                    source, "weight_pack_suffix", ".weight_packed"
                ),
                "weight_scale": getattr(source, "scale_suffix", ".weight_scale"),
            }

        rules = [("." + name, name) for name in _CANONICAL_PARAMETERS]
        for parameter, suffix in configured.items():
            if (
                not isinstance(suffix, str)
                or not suffix.startswith(".")
                or any(
                    not part or part.strip() != part for part in suffix[1:].split(".")
                )
            ):
                raise ValueError(
                    f"Invalid {parameter} checkpoint suffix {suffix!r}; expected .name"
                )
            for other_suffix, other_parameter in rules:
                if parameter != other_parameter and (
                    suffix.endswith(other_suffix) or other_suffix.endswith(suffix)
                ):
                    raise ValueError(
                        f"Ambiguous quantized checkpoint suffix {suffix!r} for {parameter}: "
                        f"overlaps {other_suffix!r} for {other_parameter}"
                    )
            rules.append((suffix, parameter))
        # A configured multi-component suffix may end in its own canonical
        # name. Prefer the longest exact suffix, without reinterpreting fields.
        self.rules = (
            tuple(sorted(set(rules), key=lambda item: len(item[0]), reverse=True))
            if configured
            else ()
        )

    def resolve(self, name):
        for suffix, parameter in self.rules:
            if name == suffix[1:]:
                return parameter
            if name.endswith(suffix):
                return name[: -len(suffix)] + "." + parameter
        return name

    def normalize(self, weights):
        if not self.rules:
            return weights
        normalized, sources = {}, {}
        for source_name, tensor in weights.items():
            name = self.resolve(source_name)
            if name in normalized:
                raise RuntimeError(
                    f"Duplicate quantized checkpoint aliases {sources[name]!r} and "
                    f"{source_name!r} both resolve to {name!r}"
                )
            normalized[name] = tensor
            sources[name] = source_name
        return normalized
