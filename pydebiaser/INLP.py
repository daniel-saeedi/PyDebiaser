import argparse
import pydebiaser, os
import wget
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer
import shutil
from pydebiaser.dataset import load_inlp_data
from pydebiaser.debias.inlp import compute_projection_matrix
from pydebiaser.model import models

class INLP:
    ''' 
    model: ["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"]
    model_name_or_path: HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.
    bias_type: ["gender", "religion", "race"]
    '''
    def __init__(self,model,model_name_or_path,bias_types,n_classifiers = 80,seed = 0):
        self.persistent_dir = os.path.dirname(pydebiaser.__file__)
        self.model = model
        self.model_name_or_path = model_name_or_path
        # self.bias_type = bias_type
        self.n_classifiers = n_classifiers
        self.seed = seed
        self.bias_types = bias_types

        print("Downloading wikipedia-2.5.txt ....")
        data_dir = self.args['persistent_dir']+"/data/text/"
        # Path(data_dir).mkdir(parents=True, exist_ok=True)
        shutil.rmtree(data_dir, ignore_errors=True)
        print(os.system("git clone https://huggingface.co/datasets/Daniel-Saeedi/wikipedia "+data_dir))
        print("Done!")
        
    
    def compute_projection_matrix(self):
        # Load data for INLP classifiers.
        data = load_inlp_data(self.persistent_dir, self.bias_type, seed=self.seed)

        # Load model and tokenizer.
        model = getattr(models, self.model)(self.model_name_or_path)
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name_or_path)

        projection_matrix = compute_projection_matrix(
            model,
            tokenizer,
            data,
            bias_type=self.bias_type,
            n_classifiers=self.n_classifiers,
        )


        os.makedirs(f"{self.persistent_dir}/results/projection_matrix", exist_ok=True)
        torch.save(
            projection_matrix,
            f"{self.persistent_dir}/results/projection_matrix/inlp_projectin_matrix.pt",
        )
        pass

    def debias(self,save = False,path = './'):
        isFirstBiasType = True

        model = None
        for bias_type in self.bias_types:
            self.bias_type = bias_type
            model = self.inlp_debias(True,path)

            if not isFirstBiasType: 
                self.args['model_name_or_path'] = path
                isFirstBiasType = False

        self.args['model_name_or_path'] = self.model_name_or_path
        return model
    #Returns the model
    def inlp_debias(self,path="./",save=False):
        self.compute_projection_matrix()

        kwargs = {}
        projection_matrix = torch.load(f"{self.persistent_dir}/results/projection_matrix/inlp_projectin_matrix.pt")
        kwargs["projection_matrix"] = projection_matrix

        # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
        model = getattr(models, 'INLP'+self.model)(self.model_name_or_path, **kwargs)
        
        if save == True:
            tokenizer2 = AutoTokenizer.from_pretrained(self.model_name_or_path)
            tokenizer2.save_pretrained(path)
            model.save_pretrained(path)

        print("Debiasing is done!")
        
        return model