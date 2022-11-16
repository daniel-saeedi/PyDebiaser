import torch
import transformers
import os
import wget
from pathlib import Path

from pydebiaser.dataset import load_sentence_debias_data
from pydebiaser.debias import (
    compute_gender_subspace,
    compute_race_subspace,
    compute_religion_subspace,
)

from transformers import AutoTokenizer

from pydebiaser.model import models
# Implementation based upon https://github.com/pliang279/sent_debias and https://github.com/McGill-NLP/bias-bench
class SentDebias:
    '''
    sent_debias_dir: sent debias codes path
    model: ["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"]
    model_name_or_path: HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.
    bias_type: ["gender", "religion", "race"]
    '''
    def __init__(self,sent_debias_dir,model,model_name_or_path,bias_type,batch_size = 32):
        self.args = {}
        self.args['persistent_dir'] = os.getcwd()
        self.args['model'] = model
        self.args['model_name_or_path'] = model_name_or_path
        self.args['bias_type'] = bias_type
        self.args['batch_size'] = batch_size

        data_dir = self.args['persistent_dir']+"/data/text/"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        filename = wget.download("https://cdn-lfs.huggingface.co/repos/d7/59/d759285e6328f18b6f31b4bad1fd24ae2073290c3eefe9b04ed144337347216d/609af0c42b9a6df2ca55a77654e0e849c8939d74fc9a3b89e7a9386138967b76?response-content-disposition=attachment%3B%20filename%3D%22wikipedia-2.5.txt%22&Expires=1668878991&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zL2Q3LzU5L2Q3NTkyODVlNjMyOGYxOGI2ZjMxYjRiYWQxZmQyNGFlMjA3MzI5MGMzZWVmZTliMDRlZDE0NDMzNzM0NzIxNmQvNjA5YWYwYzQyYjlhNmRmMmNhNTVhNzc2NTRlMGU4NDljODkzOWQ3NGZjOWEzYjg5ZTdhOTM4NjEzODk2N2I3Nj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPWF0dGFjaG1lbnQlM0IlMjBmaWxlbmFtZSUzRCUyMndpa2lwZWRpYS0yLjUudHh0JTIyIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNjY4ODc4OTkxfX19XX0_&Signature=YpeHtyhWJPncf2nSA0fMt5g-Cu9mM6R1QgKqNCarqFTucT5gFAtHl4NewRja3HdOnXKEFml-q1jMuk-~YA2~gkSnOJTaRJ78FJP024SB0iSR9Y-kBxt986fwlFMTYUicYnK3nNyXTdhbqC-~bIV3bzTwJfpkYLlsdc1j88SDKtkTAONij9oCkVDYTjanldbFxfHCCti2yrda9D2lGRcN7fX2KuDP5AYR6MOoP7PS0daPElxNGO0NdOp2aWiYjbm7yaNh2T8WB-HIyooroT69mmtBXhNuZGpAdqJeJi~xq3ZdkapkpUOa0ABWftf5h5B~MrxgH4KBIYtvbz8XDgsSqQ__&Key-Pair-Id=KVTP0A1DKRTAX", out=data_dir)
        
    # path: where do you wanna save it
    def debias(self,save = False,path = './'):
        bias_direction_path = self.compute_bias_subspace()
        
        print('Saving the debiased model...')

        kwargs = {}
        bias_direction = torch.load(bias_direction_path)
        kwargs["bias_direction"] = bias_direction
        
        if self.args['model'] == 'BertModel':
            model = 'SentenceDebiasBertModel'
        elif self.args['model'] == 'AlbertModel':
            model = 'SentenceDebiasAlbertModel'
        elif self.args['model'] == 'RobertaModel':
            model = 'SentenceDebiasRobertaModel'
        elif self.args['model'] == 'GPT2Model':
            model = 'SentenceDebiasGPT2Model'
        else:
            raise ValueError('Model not implemented')
        
        # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
        model = getattr(models, 'SentenceDebiasBertModel')(self.args['model_name_or_path'], **kwargs)
        

        if save == True:
            os.makedirs(path, exist_ok=True)
            #Saving the model
            model.save_pretrained(path)
            # Saving the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.args['model_name_or_path'])
            tokenizer.save_pretrained(path)

        return model

    def compute_bias_subspace(self):
        print("Computing bias subspace:")
        print(f" - result_dir: {self.args['persistent_dir']}")
        print(f" - model_name_or_path: {self.args['model_name_or_path']}")
        print(f" - model: {self.args['model']}")
        print(f" - bias_type: {self.args['bias_type']}")
        print(f" - batch_size: {self.args['batch_size']}")

        # Get the data to compute the SentenceDebias bias subspace.
        data = load_sentence_debias_data(
            persistent_dir=self.args['persistent_dir'], bias_type=self.args['bias_type']
        )

        # Load model and tokenizer.
        model = getattr(models, self.args['model'])(self.args['model_name_or_path'])
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.args['model_name_or_path'])

        # Specify a padding token for batched SentenceDebias subspace computation for
        # GPT2.
        if self.args['model'] == "GPT2Model":
            tokenizer.pad_token = tokenizer.eos_token
        if self.args['bias_type'] == "gender":
            bias_direction = compute_gender_subspace(
                data, model, tokenizer, batch_size=self.args['batch_size'])
        elif self.args['bias_type'] == "race":
            bias_direction = compute_race_subspace(
                data, model, tokenizer, batch_size=self.args['batch_size'])
        else:
            bias_direction = compute_religion_subspace(
                data, model, tokenizer, batch_size=self.args['batch_size'])
        
        print(f"Saving computed PCA components to: {self.args['persistent_dir']}/results/subspace/bias_direction_{self.args['bias_type']}_{self.args['model']}.pt.")
        os.makedirs(f"{self.args['persistent_dir']}/results/subspace", exist_ok=True)
        torch.save(
            bias_direction, f"{self.args['persistent_dir']}/results/subspace/bias_direction_{self.args['bias_type']}_{self.args['model']}.pt"
        )

        return f"{self.args['persistent_dir']}/results/subspace/bias_direction_{self.args['bias_type']}_{self.args['model']}.pt"

