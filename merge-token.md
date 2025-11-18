多源输入可以视为一种新的模态，通过复用现有的多模态链路，可以较简洁地实现对任意多源输入的 embedding 计算逻辑，并重用已有的 embedding cache 等能力

整体架构与流程
1. RTP-LLM 引擎从/chat/completions接口接收到带有多源输入的请求
2. RTP-LLM 前端接收到请求后，通过 Renderer 渲染得到模型推理的输入，包括推理的 prompt 和多源输入相应的模型自定义 mm_data，并将请求发送到后端
3. RTP-LLM 后端接收到 prompt 和 mm_data 后，Engine 通过 mm_data 调用 Embedding 模块，计算多源输入相应的 embedding
4. Embedding 完成计算后，返回计算得到的多源输入 mm_embedding
5. Engine 将得到的多源输入 mm_embedding，和 prompt 查表得到的 token embedding，整合成整个序列完整的 merged embedding
6. 将 merged embedding 输入模型，完成常规的自回归推理流程
7. 返回最终结果
核心主要在自定义解析和计算多源输入的 Embedding 模块上

调用栈：
FrontendServer.chat_completion(self, request: ChatCompletionRequest, raw_request: Request): rtp_llm/frontend/frontend_server.py:169
    OpenaiEndpoint.chat_completion(self, request_id: int, chat_request: ChatCompletionRequest, raw_request: Request) -> CompleteResponseAsyncGenerator: rtp_llm/openai/openai_endpoint.py:388
        CustomChatRenderer.render_chat(self, request: ChatCompletionRequest) -> RenderedInputs: rtp_llm/openai/renderers/custom_renderer.py
        grpc::Status LocalRpcServer::GenerateStreamCall(grpc::ServerContext* context, const GenerateInputPB* request, grpc::ServerWriter<GenerateOutputsPB>* writer): rtp_llm/cpp/model_rpc/LocalRpcServer.cc:122
            ErrorInfo MultimodalProcessor::updateMultimodalFeatures(std::shared_ptr<rtp_llm::GenerateInput>& input): rtp_llm/cpp/multimodal_processor/MultimodalProcessor.cc:190
                ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs, std::string ip_port = ""): rtp_llm/cpp/multimodal_processor/LocalMultimodalProcessor.h:12
                    MMEmbeddingRes.submit(self, urls: List[str], types: Optional[List[MMUrlType]] = None, tensors: Optional[List[torch.Tensor]] = None, preprocess_configs: Optional[List[List[int]]] = None): rtp_llm/utils/mm_process_engine.py:34
                        MultiModalEmbeddingInterface.mm_embedding(self, url: str, mm_type: MMUrlType, **kwargs: Any): rtp_llm/models/multimodal/multimodal_common.py:91

            注入相关参数之后会在
            GptModel::forward
                GptModel::forwardPreLayers
            这段实际使用相关数据


接入示例
模型配置
1. 修改 ckpt 的 config.json, 配置相应的特殊 token 与 embedding 模块路径
 {
   "architectures": [
     "Qwen2ForCausalLM"
   ],
   "attention_dropout": 0.0,
   ...
   "vocab_size": 217216,
+  "custom_modal": {
+    "custom_modal_token_id": 123456,
+    "embedding_module_path": "item_embedding.ItemEmbedding",
+  }
 }
2. 添加 Embedding 模块实现，组织多源输入的 embedding 计算逻辑（初步方案，具体接口与实现待定）
# demo only, subject to change

from typing import Any, List
import torch
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models.multimodal.multimodal_common import CustomMultiModalEmbeddingInterface
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.model_loader.model_weight_info import ModelWeights

class ItemEmbedding(CustomMultiModalEmbeddingInterface):
    def __init__(self,
                 config: GptInitModelParameters,
                 tokenizer: BaseTokenizer,
                 weight: ModelWeights):
        self.tokenizer = tokenizer
        self.weight = weight

    def load_weight(self):
        # get weights from LLM backbone
        self.embed_tokens = self.weight.get_global_weight("model.embed_tokens.weight")

        # load new weights
        # TBD...

    def custom_modal_embedding(self, batch_data: List[Any]) -> List[torch.Tensor]:
        # get token ids
        tokenized_data: List[torch.Tensor] = self.tokenizer.encode([
            item_part
            for data in batch_data
            for item in data
            for item_part in item
        ], padding=True, return_tensors="pt").cuda()

        batch_embedding: List[torch.Tensor] = []

        # your embedding logic here

        return batch_embedding
模型调用
通过多模态方式请求/chat/completions接口即可
 {
     "model": "...",
     "messages": [
         {
             "role": "system",
             "content": "System prompt here"
         },
         {
             "role": "user",
-            "content": "Text only user prompt here"
+            "content": [
+                {
+                    "type": "text",
+                    "text": "Multimodal user prompt here, first seq: <special_token_for_seq>, second seq: <special_token_for_seq>, third seq: <special_token_for_seq>..."
+                },
+                {
+                    "type": "custom",
+                    "data": [["sid ids1", "cat name1", "brand1"], ["sid ids2", "cat name2", "brand2"]]
+                },
+                {
+                    "type": "custom",
+                    "data": [["sid ids3", "cat name3", "brand3"], ["sid ids4", "cat name4", "brand4"], ["sid ids5", "cat name5", "brand5"]]
+                },
+                {
+                    "type": "custom",
+                    "data": [["sid ids6", "cat name6", "brand6"]]
+                }
+            ]
         }
     ],
     "extra_configs": {},
     "stream": false
 }
