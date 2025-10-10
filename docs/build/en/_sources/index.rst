RTP-LLM Documentation
====================

RTP-LLM is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor parallelism, pipeline parallelism, expert parallelism, structured outputs, chunked prefill, quantization (FP8/INT4/AWQ/GPTQ), and multi-lora batching.
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse), with easy extensibility for integrating new models.
- **Active Community**: RTP-LLM is open-source and backed by an active community with industry adoption.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   start/install.md


.. toctree::
   :maxdepth: 1
   :caption: Release Version

   release/v0.2.0/release

.. toctree::
   :maxdepth: 1
   :caption: Basic Usage

   backend/send_request.ipynb
   backend/openai_api_completions.ipynb
   backend/openai_api_vision.ipynb
   backend/openai_api_embeddings.ipynb
   backend/native_api.ipynb
   backend/cluster_envs.md

.. toctree::
   :maxdepth: 1
   :caption: Backend Tutorial

   references/deepseek/index
   references/qwen/index
   references/kimi/index


.. toctree::
   :maxdepth: 1
   :caption: Advanced Backend Configurations

   backend/server_arguments.md
   backend/sampling_params.md
   backend/attention_backend.md

.. toctree::
   :maxdepth: 1
   :caption: Supported Models

   supported_models/generative_models.md
   supported_models/multimodal_language_models.md
   supported_models/embedding_models.md
   supported_models/support_new_models.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   backend/speculative_decoding.md
   backend/reuse_kv_cache.md
   backend/function_calling.ipynb
   backend/separate_reasoning.ipynb
   backend/quantization.md
   backend/lora.ipynb
   backend/pd_disaggregation.ipynb


.. toctree::
   :maxdepth: 1
   :caption: RTP-LLM Router

   backend/flexlb.md

.. toctree::
   :maxdepth: 1
   :caption: Benchmark

   benchmark/benchmark.md

.. toctree::
      :maxdepth: 1
      :caption: References

      references/general
      references/developer


