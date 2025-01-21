# Tokre

Tokre is a token-regex library created for poking around SAE latents. It's used in the backend for https://sparselatents.com/search.

Tokre can be used for:
- Quickly creating 'synthetic features' more easily than you can with python. These are useful for searching SAE latents and quickly validating/disproving a hypothesis you have about a particular SAE latent.
- Optionally tuning a synthetic feature to try to predict a given SAE latent.

Induction ([A][B].*[A]) is literally written in tokre as
```
[a=.].*[a]
```

Taking the cosine sim of this synthetic feature with the latent activation in a gemma-2-2b SAE identifies this latent
![image](https://github.com/user-attachments/assets/696ee6ba-2285-418a-8702-0faff32878c5)


Words that start with 'M' would look like this:

```
punctuation_token = [re `[.*(:|;|,|"|\?|\.|!).*]`]
starts_with_m = [re `( m| M|M).*`]
nospace_token = [re `[^\s].*`]
m_word_token = [starts_with_m][nospace_token](?<![punctuation_token])

[m_word_token]
```

Using this script we find this latent
![image](https://github.com/user-attachments/assets/749cf5cc-5cf8-45f2-a3dc-a770f1bbe276)

Major Cities script

```
( Paris | New York | London | Tokyo | Shanghai | Dubai | Singapore | Sydney | Mumbai | Istanbul | São Paulo | Moscow | Berlin | Toronto | Seoul )
```

finds
![image](https://github.com/user-attachments/assets/8fdf930f-8447-4442-92c5-f85f308af636)


Tokre can be used to debug hypotheses and more easily find interesting exceptions:
![image](https://github.com/user-attachments/assets/9f733f0f-85fc-4c64-b716-ba65de68c408)


Tokre can easily be used with every tiktoken and huggingface tokenizer.




# Usage
```
pip install tokre
```

```
import tokre
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')

tokre.setup(tokenizer=tokenizer)

script = r'''
punctuation_token = [re `[.*(:|;|,|"|\?|\.|!).*]`]
starts_with_m = [re `( m| M|M).*`]
nospace_token = [re `[^\s].*`]
m_word_token = [starts_with_m]([nospace_token](?<![punctuation_token]))*
'''

# a 2d numpy array of token strings
# You also could have used a token ids tensor.
tok_strs = np.array([
   [' What', "'", 's', ' the', ' nicest', ' part', ' of', ' Massachusetts', '?'],
   ['Hello', ' World', '!', ' Here', "'", 's', ' an', ' example', '.']
])

synth = tokre.SynthFeat(script)
synth_acts = synth.get_mask(tok_strs)
```

