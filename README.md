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
  This work has been done as part of the <a href="https://www.fatimafellowship.com/">Fatima Fellowship</a>
</p>

<hr>



PyDebiaser is a Python package that provides 7 debiasing techniques:
- **Sent-Debias:** Extended Hard-Debias, a word embedding debiasing technique. You can try out this technique [here](https://huggingface.co/spaces/Daniel-Saeedi/sent-debias). [Paper](https://arxiv.org/abs/2007.08100)
- **Self-Debias:** leverages a model’s internal knowledge to discourage it from generating biased text. [Paper](https://arxiv.org/abs/2103.00453)
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

## Top-k
Top-k is an inference text generation method that generates k different texts up to l tokens in length from a prompt and then select the least toxic text using Detoxify.The text generator will be prompted again by the selected text from the previous step to generate the subsequent k text completion with the same length and then choose the best completion. This process continues until we have the desired text length. More importantly, unlike other techniques, the Top-k debiaser does not reduce the language model score.


Run this code to debias:
```
from pydebiaser.TopK import TopK
debiaser = TopK(model,tokenizer)
debiaser.generate(prompt,max_len,step = l,number_of_samples = k) # By default, l = 25 and k = 10
```

**Example**
```
from pydebiaser.TopK import TopK
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

debiaser = TopK(model,tokenizer)
debiaser.generate("Two muslims walked into a bar",200,step = 25,number_of_samples = 10)
```

Top-k output:

```
Two muslims walked into a bar and were asked about their religion, and the majority, only half were told to "wait and see" when asked how they had learned to pray.

In May last year, a British woman claimed that she had learned English while she was a teenager after having her mother, a young man, take her to a mosque for training.

She told the BBC that she "had no idea what was going on", and that she was shocked by how "very few people knew that there was a difference between what they did and what we did."

A separate study found that more people in the country knew that their religion was 'a bit different' than other religions, and that the number of people with different religious identities jumped from 50,000 to almost 50,000 in four years.

The findings came from a survey of 3,800 people aged between 15 and 65 at London's West End.

They showed that the number
```

Original GPT-2 generation:
```
Two muslims walked into a bar and had a drink, but they did not return."\n\nA man approached the bar\'s door. 

The man then told the Muslim youth that he would leave if he went in. The youth later returned and told him to stay.

The young man reported him to the police. He said he saw four people walking past and that they were Arabs. When he asked where the Arabs were, he said people were sitting between them. He said he did not know where those Arabs were but would return. 

He told the police he found the Arabs in his car and was able to get to them in a car carrying three people. He told the police that when he reached the area he saw that the "residents" were walking past and also that he heard the Palestinians screaming from other Palestinian areas over the loudspeaker. He also said he looked at the men and saw several Muslims praying and saying they would give him the money.\n\nAs reported by
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



