import argparse
import pydebiaser, os
import wget
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer
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
    def __init__(self,model,model_name_or_path,bias_type,n_classifiers = 80,seed = 0):
        self.persistent_dir = os.path.dirname(pydebiaser.__file__)
        self.model = model
        self.model_name_or_path = model_name_or_path
        self.bias_type = bias_type
        self.n_classifiers = n_classifiers
        self.seed = seed

        print("Downloading wikipedia-2.5.txt ....")
        data_dir = self.persistent_dir+"/data/text/"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        filename = wget.download("https://cdn-lfs.huggingface.co/repos/d7/59/d759285e6328f18b6f31b4bad1fd24ae2073290c3eefe9b04ed144337347216d/609af0c42b9a6df2ca55a77654e0e849c8939d74fc9a3b89e7a9386138967b76?response-content-disposition=attachment%3B%20filename%3D%22wikipedia-2.5.txt%22&Expires=1668878991&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZG4tbGZzLmh1Z2dpbmdmYWNlLmNvL3JlcG9zL2Q3LzU5L2Q3NTkyODVlNjMyOGYxOGI2ZjMxYjRiYWQxZmQyNGFlMjA3MzI5MGMzZWVmZTliMDRlZDE0NDMzNzM0NzIxNmQvNjA5YWYwYzQyYjlhNmRmMmNhNTVhNzc2NTRlMGU4NDljODkzOWQ3NGZjOWEzYjg5ZTdhOTM4NjEzODk2N2I3Nj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPWF0dGFjaG1lbnQlM0IlMjBmaWxlbmFtZSUzRCUyMndpa2lwZWRpYS0yLjUudHh0JTIyIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNjY4ODc4OTkxfX19XX0_&Signature=YpeHtyhWJPncf2nSA0fMt5g-Cu9mM6R1QgKqNCarqFTucT5gFAtHl4NewRja3HdOnXKEFml-q1jMuk-~YA2~gkSnOJTaRJ78FJP024SB0iSR9Y-kBxt986fwlFMTYUicYnK3nNyXTdhbqC-~bIV3bzTwJfpkYLlsdc1j88SDKtkTAONij9oCkVDYTjanldbFxfHCCti2yrda9D2lGRcN7fX2KuDP5AYR6MOoP7PS0daPElxNGO0NdOp2aWiYjbm7yaNh2T8WB-HIyooroT69mmtBXhNuZGpAdqJeJi~xq3ZdkapkpUOa0ABWftf5h5B~MrxgH4KBIYtvbz8XDgsSqQ__&Key-Pair-Id=KVTP0A1DKRTAX", out=data_dir)
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

    #Returns the model
    def debias(self,path="./",save=False):
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