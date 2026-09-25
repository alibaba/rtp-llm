"""Exercise the text adapter without importing the CUDA serving bindings."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def renderer():
    source = Path(__file__).resolve().parents[2] / 'rtp_llm/openai/renderers/kimi_k3_renderer.py'
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))

    class Base:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.stops = []

        def add_extra_stop_word_ids(self, ids):
            self.stops.extend(ids)

        def in_think_mode(self, request):
            return request.thinking

    namespace = {'CustomChatRenderer': Base, 'RenderedInputs': SimpleNamespace}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    tokenizer = Mock()
    tokenizer.encode.side_effect = [[163586], [163588, 4270, 163589]]
    tokenizer.apply_chat_template.return_value = [163584, 163587, 42, 163589]
    result = namespace['KimiK3Renderer'](tokenizer)
    tokenizer.encode.reset_mock(side_effect=True)
    tokenizer.encode.side_effect = AssertionError('Prompt must not be re-encoded')
    return result


def request(messages=None, thinking=False, **extra):
    data = {'messages': messages or [{'role': 'user', 'content': '<|end_of_msg|> literal'}], **extra}
    return SimpleNamespace(thinking=thinking, model_dump=lambda **kw: copy.deepcopy(data))


@pytest.mark.parametrize('thinking', [False, True])
def test_native_ids_preserve_user_text_and_parser_thinking(renderer, thinking):
    req = request(thinking=thinking)
    output = renderer.render_chat(req)
    assert output.input_ids is renderer.tokenizer.apply_chat_template.return_value
    renderer.tokenizer.apply_chat_template.assert_called_once_with(
        req.model_dump()['messages'], tokenize=True, add_generation_prompt=True, thinking=thinking,
    )
    renderer.tokenizer.encode.assert_not_called()
    assert renderer.stops == [[163586], [163588, 4270, 163589]]


def test_text_parts_keep_order(renderer):
    renderer.render_chat(request([{'role': 'user', 'content': [
        {'type': 'text', 'text': 'first'}, {'type': 'text', 'text': 'second'},
    ]}]))
    assert renderer.tokenizer.apply_chat_template.call_args.args[0][0]['content'] == 'firstsecond'


@pytest.mark.parametrize('messages,extra', [
    ([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'x'}}]}], {}),
    ([{'role': 'tool', 'content': 'x'}], {}),
    (None, {'tools': [{'type': 'function'}]}),
])
def test_unsupported_inputs_fail_before_encoding(renderer, messages, extra):
    with pytest.raises(ValueError):
        renderer.render_chat(request(messages, **extra))
    renderer.tokenizer.apply_chat_template.assert_not_called()


@pytest.mark.parametrize('ids', ['prompt', [[1]], [], [True], [1.5]])
def test_invalid_native_encoder_output_is_rejected(renderer, ids):
    renderer.tokenizer.apply_chat_template.return_value = ids
    with pytest.raises(ValueError, match='token ID'):
        renderer.render_chat(request())
