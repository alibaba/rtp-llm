from transformers import BloomTokenizerFast
from tokenizers import pre_tokenizers

class Tokenizer13B(BloomTokenizerFast):

    def __init__(
            self,
            do_lower_case=False,
            remove_space=False,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            model_max_length=2048,
            **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            model_max_length=model_max_length,
            **kwargs
        )

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        self.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(True),
            self.backend_tokenizer.pre_tokenizer
        ])
