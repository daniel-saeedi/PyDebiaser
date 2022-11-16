import re
import json
'''
    Character Neutralization
    We replace words that the model is biased towards them with the words like person and people
    to prevent generating biased prompts
'''
class CharacterNeutralization:
    def __init__(self,data = 'data/character_neutralization.json'):
        f = open(data)
        self.roles = json.load(f)

    # This method returns anti stereotype 
    def neutralize(self,prompt):
        for word in self.roles:
            regex = r"\b" + re.escape(word.lower()) + r"\b"
            prompt = re.sub(regex, self.roles[word].lower(), prompt.lower())

            regex = r"\b" + re.escape(word.lower()) +"s" + r"\b"
            prompt = re.sub(regex, self.roles[word].lower()+"s", prompt.lower())
        
        return prompt

    def reverse_neutralize(self,prompt):
        for word in self.roles:
            word = word.lower()
            regex = r"\b" + re.escape(self.roles[word]) + r"\b"
            prompt = re.sub(regex, word, prompt.lower())
        
        return prompt

    # If you set swap_whole_text = False, then only the initial prompt will be changed
    def generate(self,model,tokenizer,prompt,maxlen_gen,swap_whole_text = True):
        prompt_swapped = self.neutralize(prompt)
        prompt_swapped_len = len(prompt_swapped)

        # print(prompt_swapped)
        inputs = tokenizer(prompt_swapped, return_tensors="pt")

        generation = model.generate(inputs["input_ids"],max_length=maxlen_gen, do_sample=True, top_k=50, top_p=0.9,num_return_sequences = 1)[0]

        text = tokenizer.decode(generation)

        return prompt + tokenizer.decode(generation)[prompt_swapped_len:]