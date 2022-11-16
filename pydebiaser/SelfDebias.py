import numpy as np
from pydebiaser.debias.self_debias.modeling import SelfDebiasGenerativeLM
import pydebiaser
import os
from pydebiaser.model import models
class SelfDebias:
    '''
    model: ["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"]
    model_name_or_path: HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.
    bias_type: ["gender", "religion", "race"]
    '''
    def __init__(self,model,model_name_or_path,bias_type):
        self.args = {}
        self.args['persistent_dir'] = os.path.dirname(pydebiaser.__file__)
        self.args['model'] = model
        self.args['model_name_or_path'] = model_name_or_path
        self.args['bias_type'] = bias_type

        if self.args['model'] == 'BertModel':
            m = 'SelfDebiasBertForMaskedLM'
        elif self.args['model'] == 'AlbertModel':
            m = 'SelfDebiasAlbertForMaskedLM'
        elif self.args['model'] == 'RobertaModel':
            m = 'SelfDebiasRobertaForMaskedLM'
        elif self.args['model'] == 'GPT2Model':
            m = 'SelfDebiasGPT2LMHeadModel'
        else:
            raise ValueError('Model not implemented')
        # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
        # kwargs = {}
        # kwargs["model_name_or_path"] = self.args['model_name_or_path']
        self_debias_m = getattr(models, m)(model_name_or_path)

        self.model = SelfDebiasGenerativeLM(model_class=self_debias_m, model_name_or_path=self.args['model_name_or_path'], use_cuda=False)
    
    def generate(self,prompt,max_len):
        debiased_output = self.model.generate_self_debiasing([prompt], debiasing_prefixe=self.args['bias_type'], max_length=max_len)

        return prompt + debiased_output
    