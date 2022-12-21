import torch
import transformers
import pydebiaser, os
import shutil
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
    model: ["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"]
    model_name_or_path: HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.
    bias_type: ["gender", "religion", "race"]
    '''
    def __init__(self,model,model_name_or_path,bias_types,batch_size = 32):
        self.args = {}
        self.args['persistent_dir'] = os.path.dirname(pydebiaser.__file__)
        self.args['model'] = model
        self.args['model_name_or_path'] = model_name_or_path
        # self.args['bias_type'] = bias_type
        self.args['batch_size'] = batch_size
        self.bias_types = bias_types
        self.model_name_or_path = model_name_or_path

        print("Downloading wikipedia-2.5.txt ....")
        data_dir = self.args['persistent_dir']+"/data/text/"
        # Path(data_dir).mkdir(parents=True, exist_ok=True)
        shutil.rmtree(data_dir, ignore_errors=True)
        print(os.system("git clone https://huggingface.co/datasets/Daniel-Saeedi/wikipedia "+data_dir))
        
        print("Done!")

    
    def debias(self,save = False,path = './'):
        isFirstBiasType = True

        model = None
        for bias_type in self.bias_types:
            self.args['bias_type'] = bias_type
            model = self.sent_debias(True,path)

            if not isFirstBiasType: 
                self.args['model_name_or_path'] = path
                isFirstBiasType = False

        self.args['model_name_or_path'] = self.model_name_or_path
        return model
    # path: where do you wanna save it
    def sent_debias(self,save = False,path = './'):
        print('Computing bias subspaces...')
        bias_direction_path = self.compute_bias_subspace()
        
        print('Saving the debiased model...')

        kwargs = {}
        bias_direction = torch.load(bias_direction_path)
        kwargs["bias_direction"] = bias_direction
        
        m = ''
        if self.args['model'] == 'BertModel':
            m = 'SentenceDebiasBertModel'
        elif self.args['model'] == 'AlbertModel':
            m = 'SentenceDebiasAlbertModel'
        elif self.args['model'] == 'RobertaModel':
            m = 'SentenceDebiasRobertaModel'
        elif self.args['model'] == 'GPT2Model':
            m = 'SentenceDebiasGPT2Model'
        else:
            raise ValueError('Model not implemented')
        
        # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
        model = getattr(models, m)(self.args['model_name_or_path'], **kwargs)
        

        if save == True:
            os.makedirs(path, exist_ok=True)
            #Saving the model
            model.save_pretrained(path)
            # Saving the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.args['model_name_or_path'])
            tokenizer.save_pretrained(path)
        
        print("Debiasing is done!")
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

