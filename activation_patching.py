from IPython.display import clear_output

from nnsight import CONFIG
from nnsight import LanguageModel, util
from nnsight.tracing.graph import Proxy

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "iframe"

# Load Model (gpt2)
model = LanguageModel("openai-community/gpt2", device_map="auto")
clear_output()
# Model has 12 layers
print(model)

clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = (
    "After John and Mary went to the store, John gave a bottle of milk to"
)

# Use a tokenizer on "John" and "Mary" to find the tokens that represent them.
# We are looking for the whitespace+word token
correct_index = model.tokenizer(" John")['input_ids'][0]
incorrect_index = model.tokenizer(" Mary")['input_ids'][0]
print(f"' John': {correct_index}")
print(f"' Mary': {incorrect_index}")

# Step 1 : Clean Run
# We run the model with the clean prompt, collect the final output of each layer, and record the logit difference
# between correct answer token ( John) and wrong answer token ( Mary)

N_LAYERS = len(model.transformer.h)

with model.trace(clean_prompt) as tracer:
    clean_tokens = tracer.invoker.inputs[0][0]['input_ids'][0]

    # Get hidden states of all layers in the network.
    # We index the output at 0 because it's a tuple where the first index is the hidden state.

    clean_hs = [
        model.transformer.h[layer_idx].output[0].save()
        for layer_idx in range(N_LAYERS)
    ]

    # Get logits from the lm_head.
    clean_logits = model.lm_head.output

    # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
    clean_logit_diff = (
        clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]
    ).save()

# Step 2 : Corrupted Run
# We run the model with the corrupted prompt, collect the final output of each layer, and record the logit difference
# between correct answer token ( John) and wrong answer token ( Mary)

with model.trace(corrupted_prompt) as tracer:
    corrupted_logits = model.lm_head.output

    # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
    corrupted_logit_diff = (
        corrupted_logits[0, -1, correct_index] - corrupted_logits[0, -1, incorrect_index]
    ).save()

# Step 3: Activation Patching Intervention
# For each token position in the clean prompt, loop through all layers of the model
# In each layer, run a forward pass using the corrupted prompt, and patch the activation from the clean run 
# Collect the final output difference between the correct and incorrect answer tokens for each patched activation

ioi_patching_results = []

# Iterate through all the layers
for layer_idx in range(len(model.transformer.h)):
    _ioi_patching_results = []

    # Iterate through all tokens
    for token_idx in range(len(clean_tokens)):
        # Patching corrupted run at given layer and token
        with model.trace(corrupted_prompt) as tracer:
            # Apply the patch from the clean hidden states to the corrupted hidden states.
            model.transformer.h[layer_idx].output[0][:, token_idx, :] = clean_hs[layer_idx][:, token_idx, :]

            patched_logits = model.lm_head.output

            patched_logit_diff = (
                patched_logits[0, -1, correct_index]
                - patched_logits[0, -1, incorrect_index]
            )

            # Calculate the improvement in the correct token after patching.
            patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                clean_logit_diff - corrupted_logit_diff
            )

            _ioi_patching_results.append(patched_result.item().save())

    ioi_patching_results.append(_ioi_patching_results)

# Visualise Results
def plot_ioi_patching_results(model,
                              ioi_patching_results,
                              x_labels,
                              plot_title="Normalized Logit Difference After Patching Residual Stream on the IOI Task"):

    ioi_patching_results = util.apply(ioi_patching_results, lambda x: x.value, Proxy)

    fig = px.imshow(
        ioi_patching_results,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Position", "y": "Layer","color":"Norm. Logit Diff"},
        x=x_labels,
        title=plot_title,
    )

    return fig

print(f"Clean logit difference: {clean_logit_diff:.3f}")
print(f"Corrupted logit difference: {corrupted_logit_diff:.3f}")

clean_decoded_tokens = [model.tokenizer.decode(token) for token in clean_tokens]
token_labels = [f"{token}_{index}" for index, token in enumerate(clean_decoded_tokens)]

fig = plot_ioi_patching_results(model, ioi_patching_results,token_labels,"Patching GPT-2-small Residual Stream on IOI task")
fig.show()