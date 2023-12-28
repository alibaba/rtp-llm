import json
from typing import Any, Dict, List, Union, Tuple

from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.config.generate_config import RequestFormat

class MultiModalMixin:
    @staticmethod
    def process_encode_plugin(prompt: str, generate_config: Dict[str, Any], special_tokens: Any, tokenizer: Any, **kwargs: Any) -> List[int]:
        if len(prompt) == 0:
            raise FtRuntimeException(ExceptionType.EMPTY_PROMPT_ERROR, "prompt should have at least one token!")
        if type(prompt) is not str:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "expect string prompt, actual: " + str(prompt))
        return tokenizer.encode(prompt)

    @staticmethod
    def multimodal_modify_prompt_plugin(prompt: Union[List[Dict[str, Any]], str], **kwargs: Any) -> Tuple[str, List[Any]]:
        img_token: str = kwargs.get('img_token')
        if kwargs.get('generate_config', {})['request_format'] == RequestFormat.CHAT_API:
            if isinstance(prompt, str):
                messages = json.loads(prompt, strict=False)
            else:
                messages = prompt
            new_prompt: str = ""
            new_images: List[str] = []
            for message in messages:
                new_prompt += message['role'].upper() + ' :'
                if isinstance(message['content'], str):
                    new_prompt += message['content'] + '\n'
                elif isinstance(message['content'], List):
                    for x in message['content']:
                        if x['type'] == 'text':
                            new_prompt += x['text']
                        elif x['type'] == 'image_url':
                            now_images = x['image_url']
                            if isinstance(now_images, List):
                                new_images.extend(now_images)
                                new_prompt += (img_token + '\n') * len(now_images)
                            else:
                                new_images.append(now_images)
                                new_prompt += img_token + '\n'
                        else:
                            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "content type can only be text or image_url, but get: " + x['type'])
                    new_prompt += '\n'
            return new_prompt + 'ASSISTANT :', new_images
        elif isinstance(prompt, List):
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, "raw request format cannot accept dict prompt")
        return prompt, kwargs['image']
