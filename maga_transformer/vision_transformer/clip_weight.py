class ClipVitWeights:
    def __init__(self, vit):
        self.hf_prefix = "transformer.visual."
        self.ft_prefix = "self.visual."
        self.weight_names = self._get_vit_params(vit)
    
    def _get_vit_params(self, vit):
        return list(vit.state_dict().keys())
