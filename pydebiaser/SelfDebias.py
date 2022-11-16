import numpy as np
import pydebiaser
import os
from transformers import GPT2LMHeadModel
from pydebiaser.self_debias.modeling import SelfDebiasGenerativeLM

class SelfDebias:
    '''
    model: ["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"]
    model_name_or_path: HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.
    bias_type: ["gender", "religion", "race"]
    '''
    def __init__(self,model_name_or_path,bias_type):
        self.model_name_or_path = model_name_or_path
        self.bias_type = bias_type
        self.model = SelfDebiasGenerativeLM(model_class=GPT2LMHeadModel, model_name_or_path=self.model_name_or_path, use_cuda=False)

    
    def generate(self,prompt,max_len):
        debiased_output = self.model.generate_self_debiasing([prompt], debiasing_prefixe=self.bias_type, max_length=max_len)

        return prompt + ' '.join(debiased_output)
    