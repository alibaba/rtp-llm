"""Direct Qwen4 target integration with native execution views.

The public parameter tree owns checkpoint loading. The existing main model is
a borrowed execution view, not another state_dict or legacy ModelWeights map.
"""
from dataclasses import replace
from rtp_llm.models_py.new_models.qwen4_exp.target_weights import Qwen4TargetWeights


class Qwen4ForCausalLM(Qwen4TargetWeights):
    def __init__(self, model_config, load_config, *, checkpoint_plan=None):
        # The frontend must use main's resolved/frozen Qwen4 deployment config.
        # Raw JSON is only accepted by the checkpoint-only CPU owner.
        if isinstance(model_config,dict) or load_config.parallelism_config is None:
            raise ValueError("Qwen4 runtime requires its resolved model and parallelism configs")
        from rtp_llm.config.qwen4_exp_config import freeze_runtime_config, resolve_qwen4_parallelism, validate_qwen4_runtime
        parallelism = resolve_qwen4_parallelism(model_config,load_config.parallelism_config)
        load_config = replace(load_config,parallelism_config=parallelism,
            ffn_tp_size=parallelism.get_ffn_tp_size(),ffn_tp_rank=parallelism.get_ffn_tp_rank())
        validate_qwen4_runtime(model_config,parallelism)
        freeze_runtime_config(model_config,parallelism)
        super().__init__(model_config,load_config,checkpoint_plan=checkpoint_plan)
        self.config = model_config
        self.load_config = load_config
        object.__setattr__(self,"_runtime",None)

    def process_weights_after_loading(self):
        if self._runtime is not None:
            return
        # Finalize checkpoint consumers before constructing the borrowed runtime.
        # Hooks are idempotent when the loader subsequently visits each child.
        for module in reversed(tuple(self.modules())):
            if module is not self:
                hook = getattr(module,"process_weights_after_loading",None)
                if hook is not None:
                    hook()
        from rtp_llm.models_py.new_models.qwen4_exp.runtime_bindings import bind_target
        runtime = bind_target(self,self.config,self.load_config)
        for parameter in self.parameters():
            parameter._rtp_requires_restart_for_update = True
        object.__setattr__(self,"_runtime",runtime)

    def __getattr__(self,name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            runtime = self.__dict__.get("_runtime")
            if runtime is None:
                raise
            return getattr(runtime,name)

    def forward(self,*args,**kwargs):
        if self._runtime is None:
            raise RuntimeError("Qwen4 target weights/runtime are not finalized")
        return self._runtime(*args,**kwargs)

    @property
    def kv_cache(self):
        runtime = self.__dict__.get("_runtime")
        return None if runtime is None else runtime.kv_cache

    @kv_cache.setter
    def kv_cache(self, cache):
        runtime = self.__dict__.get("_runtime")
        if runtime is None:
            raise RuntimeError("Finalize Qwen4 runtime before installing its KV cache")
        runtime.kv_cache = cache

    def runtime_weight_view(self):
        weights = {"embedding":self.embed_tokens.weight,"lm_head":self.lm_head.weight}
        if self.embed_tokens.storage_scale is not None:
            weights["embedding.qwen4_fp8_storage_scale"] = self.embed_tokens.storage_scale
        if self.lm_head.storage_scale is not None:
            weights["lm_head.weight_only_quant_scale"] = self.lm_head.storage_scale
        return weights

    def close(self):
        runtime = self.__dict__.get("_runtime")
        if runtime is not None:
            runtime.close()
        else:
            super().close()
