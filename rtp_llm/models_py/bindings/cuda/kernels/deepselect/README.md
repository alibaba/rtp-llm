# DeepSelect FP32 K4096

FP32 header subset from alibaba/rtp-llm commit
`75040a35b6e38c2554e19faf9c43b1c320715955`, retaining the accompanying LICENSE.
The wrapper instantiates only K4096 values-and-indices selection on SM100/SM103.
Other devices, shapes, dtypes and K values use the ATen fallback.
