class ClipVitWeights:
    def __init__(self, vit):
        self.hf_prefix = "transformer.visual."
        self.ft_prefix = "self.visual."
        self.weight_names = list(vit.state_dict().keys())
