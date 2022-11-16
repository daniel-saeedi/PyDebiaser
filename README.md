<p align="center">
  <img src="https://raw.githubusercontent.com/daniel-saeedi/PyDebiaser/main/PyDebiaser.png" />
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
> !cd /content/PyDebiaser && pip install .

## How to use
Run this command to install PyDebiaser:
> !git clone https://github.com/daniel-saeedi/PyDebiaser.git
> 
> !cd /content/PyDebiaser && pip install .
