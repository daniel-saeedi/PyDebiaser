import numpy as np
from detoxify import Detoxify

from pydebiaser.BiasSwapping import BiasSwapping
from pydebiaser.PrependAdj import PrependAdj
from pydebiaser.CharacterNeutralization import CharacterNeutralization

class TopK:
    def __init__(self,model,tokenizer):
        self.classifier_unbiased = Detoxify('unbiased')
        self.model = model
        self.tokenizer = tokenizer
        # self.bias_swap = BiasSwapping()
        # self.prepend_adj = PrependAdj()
        # self.character_neutralization = CharacterNeutralization()

    '''
    * max_len: The length of generated text
    * step: indicates how many tokens are added to the text in each step
    * number_of_samples: indicates how many samples are generated in each step
    '''
    def generate(self,original_prompt,max_len,step = 25,number_of_samples = 10,bias_swapping = False,prepend_adj = False,character_neutralization = False):
        prompt = original_prompt
        if bias_swapping:
            prompt = self.bias_swap.swap_biases(prompt)
        if character_neutralization:
            prompt = self.character_neutralization.neutralize(prompt)
        if prepend_adj:
            prompt = self.prepend_adj.prepend(prompt)
        
        prompt_length = len(prompt)
        
        for i in range(1,max_len//step + 1):
            maxlen_gen = 0
            if i == max_len//step:
                maxlen_gen = max_len
            else:
                maxlen_gen = step*i
            
            
            inputs = self.tokenizer(prompt, return_tensors="pt")

            generations = self.model.generate(inputs["input_ids"],max_length=maxlen_gen, do_sample=True, top_k=50, top_p=0.9,num_return_sequences = number_of_samples)

            bias_score = []

            for s in generations:
                generated = self.tokenizer.decode(s)
                
                b_score = self.classifier_unbiased.predict(generated[len(prompt):])
                # bias_score.append(sum(b_score.values()))
                bias_score.append(b_score['toxicity'])
            
            prompt = self.tokenizer.decode(generations[np.array(bias_score).argmin()])

        if bias_swapping:
            return original_prompt + self.bias_swap.reverse_swap_biases(prompt[prompt_length:])
        else:
            return original_prompt + prompt[prompt_length:]