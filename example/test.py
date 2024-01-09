
from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory

if __name__ == '__main__':
    model = ModelFactory.from_huggingface("Qwen/Qwen-7B-Chat")
    pipeline = Pipeline(model, model.tokenizer)
    for res in pipeline(["<|im_start|>user\nhello, what's your name<|im_end|>\n<|im_start|>assistant\n"], max_new_tokens = 100):
        print(res.batch_response)
    pipeline.stop()
