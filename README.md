<p align="center">
  <img src="https://raw.githubusercontent.com/daniel-saeedi/PyDebiaser/main/PyDebiaser.png" />
  <br><br>
</p>

<p align="center">
  <b>Contributors:</b> <a href="https://daniel-saeedi.github.io/">Daniel Saeedi</a> and <a href="https://github.com/kunwarsaaim/">Kunwar M. Saaim</a>
</p>

<p align="center">
  <b>Mentor:</b> <a href="https://abidlabs.github.io/">Abubakar Abid</a>
</p>

<p align="center">
  Part of the <a href="https://www.fatimafellowship.com/">Fatima Fellowship</a> research
</p>

<hr>



PyDebiaser is a Python package that provides 7 debiasing techniques:
- **Self-Debias:** leverages a model’s internal knowledge to discourage it from generating biased text. [Paper]([https://arxiv.org/abs/2007.08100](https://arxiv.org/abs/2103.00453))
- **Sent-Debias:** Extended Hard-Debias, a word embedding debiasing technique. [Paper](https://arxiv.org/abs/2007.08100)
- **INLP:** Debiases a model’s representations by training a linear classifier to predict the protected property to remove (e.g., gender). [Paper](https://arxiv.org/abs/2004.07667)
- **Top-k:** Generates k different texts and selects the least toxic one using Detoxifier.
- **Bias-Swapping:** This technique swaps words pairs such as "she" with "he", "muslim" with "christian", and etc in the prompt.
- **Prepend-Adjective:** This technique prepends positive adjective before a biased words such as "woman" in the prompt(E.g. successful woman).
- **Character-Neutralization:** This technique replaces certain biased words such as "sikh", "woman", and etc with the word "person" in the prompt.

## Installation
Run this command to install PyDebiaser:
> !git clone https://github.com/daniel-saeedi/PyDebiaser.git
> 
> !cd PyDebiaser && pip install .

## Self-Debias
Run this code to debias:
```
from pydebiaser.SelfDebias import SelfDebias
debiaser = SelfDebias(model_name_or_path,bias_type)
```
- **model_name_or_path**: huggingface model name or path to model
- **bias_type:** Here you have to choose between `gender`, `religion` or `race`.

**Note:** At this moment, Self-Debias has been implemented for GPT-2 only. You can use any pre-trained GPT-2 model.

And finally generate a text using the following code:
```
debiaser.generate(prompt,max_len)
```


## INLP
Run this code to debias:
```
from pydebiaser.INLP import INLP
debiaser = INLP('BertModel','bert-base-uncased','gender')
debiaser = INLP(model,model_name_or_path,bias_type)
```
- **model names:** ["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"]
- **model_name_or_path**: huggingface model name or path to model
- **bias_type:** Here you have to choose between `gender`, `religion` or `race`.

**Note:** You can debias any pretrained Bert, Albert, Robert, or GPT2 like models.

And finally debias the model using the following code:
```
model = debiaser.debias(save=True,path = '/content/result/debiased/')
```
- debias method returns the debiased model. 
- Optional: `save` and `path` parameters are used for saving the model.

## Sent-Debias
Run this code to debias:
```
from pydebiaser.SentDebias import SentDebias
debiaser = SentDebias('BertModel','bert-base-uncased','gender')
debiaser = SentDebias(model,model_name_or_path,bias_type)
```
- **model names:** ["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"]
- **model_name_or_path**: huggingface model name or path to model
- **bias_type:** Here you have to choose between `gender`, `religion` or `race`.

**Note:** You can debias any pretrained Bert, Albert, Robert, or GPT2 like models.

And finally debias the model using the following code:
```
model = debiaser.debias(save=True,path = '/content/result/debiased/')
```
- debias method returns the debiased model. 
- Optional: `save` and `path` parameters are used for saving the model.

## Sent-Debias
Run this code to debias:
```
from pydebiaser.SentDebias import SentDebias
debiaser = SentDebias('BertModel','bert-base-uncased','gender')
debiaser = SentDebias(model,model_name_or_path,bias_type)
```
- **model names:** ["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"]
- **model_name_or_path**: huggingface model name or path to model
- **bias_type:** Here you have to choose between `gender`, `religion` or `race`.

**Note:** You can debias any pretrained Bert, Albert, Robert, or GPT2 like models.

And finally debias the model using the following code:
```
model = debiaser.debias(save=True,path = '/content/result/debiased/')
```
- debias method returns the debiased model. 
- Optional: `save` and `path` parameters are used for saving the model.

## Top-k
Run this code to debias:
```
from pydebiaser.TopK import TopK
debiaser = TopK(model,tokenizer)
```

And finally generate a text using the following code:
```
debiaser.generate(prompt,max_len)
```

## BiasSwapping
```
from pydebiaser.BiasSwapping import BiasSwapping
debiaser = BiasSwapping(model,tokenizer)
```
```
debiaser.generate(prompt,max_len)
```

## CharacterNeutralization
```
from pydebiaser.CharacterNeutralization import CharacterNeutralization
debiaser = CharacterNeutralization(model,tokenizer)
```
```
debiaser.generate(prompt,max_len)
```

## PrependAdj
```
from pydebiaser.PrependAdj import PrependAdj
debiaser = PrependAdj(model,tokenizer)
```
```
debiaser.generate(prompt,max_len)
```



