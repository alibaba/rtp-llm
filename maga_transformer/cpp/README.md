cpp重构 设计文档

# MagaOp

This class defines the boundary between c++ and python.
It takes tokenized requests as input and returns an asynchronuously updated query as output.

It implements basic query execution loop:
*get query batch -> assemble -> run model -> update output*

Besides, it deals with params from python world such as weights,
converts them to internal data structures,
thus making internal components isolated with torch dependencies.

# components

These are components used in the main execution loop, including:
 - QueryManager: maintains query queue and batch queries
 - QueryAssembler: assembles query batch into executor request
 - Executor: calls model implementation based on assembled queries

# model

These are models implementations calling hardware computational API at `src/fastertransformer/devices`

currently we have:
 - GptModel: corresponding to `PreTrainedModel` in transformers
 - Sampler: an all-in-one sampling layer

