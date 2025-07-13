from IPython.display import clear_output
from nnsight import LanguageModel
from typing import List, Callable
import torch
import numpy as np
from IPython.display import clear_output

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "iframe"

clear_output()

# Load model (gpt2)
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

# 12 layers
print(model)

# Logit Lens:
# We apply a softmax layer to each layer's output to gain a glimpse into what output the model is 
# predicting at each processing step

# Apply Logit Lens
# 1. Collect activations at each layer's output
# 2. Apply normalization
# 3. Process through the model's head to get logits
# 4. Apply the softmax to the logits to obtain output token probabilities.

prompt= "The Eiffel Tower is in the city of"
layers = model.transformer.h
probs_layers = []

with model.trace() as tracer:
    with tracer.invoke(prompt) as invoker:
        for layer_idx, layer in enumerate(layers):
            # Process layer output through the model's head and layer normalization
            layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))

            # Apply softmax to obtain probabilities and save the result
            probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
            probs_layers.append(probs)

probs = torch.cat([probs.value for probs in probs_layers])

# Find the maximum probability and corresponding tokens for each position
max_probs, tokens = probs.max(dim=-1)

# Decode token IDs to words for each layer
words = [[model.tokenizer.decode(t.cpu()).encode("unicode_escape").decode() for t in layer_tokens]
    for layer_tokens in tokens]

# Access the 'input_ids' attribute of the invoker object to get the input words
input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]]

# Visualising GPT-2 Layer Interpretations
fig = px.imshow(
    max_probs.detach().cpu().numpy(),
    x=input_words,
    y=list(range(len(words))),
    color_continuous_scale=px.colors.diverging.RdYlBu_r,
    color_continuous_midpoint=0.50,
    text_auto=True,
    labels=dict(x="Input Tokens", y="Layers", color="Probability")
)

fig.update_layout(
    title='Logit Lens Visualization',
    xaxis_tickangle=0
)

fig.update_traces(text=words, texttemplate="%{text}")
fig.show()