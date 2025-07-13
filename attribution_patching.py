from IPython.display import clear_output
from nnsight import LanguageModel
import einops
import torch
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "iframe"

# Load model (GPT-2)
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
clear_output()
print(model)

# Step 1: IOI Patching
prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]

answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

clean_tokens = model.tokenizer(prompts, return_tensors="pt")["input_ids"]
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]

answer_token_indices = torch.tensor(
    [[model.tokenizer(answers[i][j])["input_ids"][0] for j in range(2)]
     for i in range(len(answers))]
)

def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

# Base logits
clean_logits = model.trace(clean_tokens, trace=False).logits.cpu()
corrupted_logits = model.trace(corrupted_tokens, trace=False).logits.cpu()

CLEAN_BASELINE = get_logit_diff(clean_logits).item()
CORRUPTED_BASELINE = get_logit_diff(corrupted_logits).item()

def ioi_metric(logits):
    return (get_logit_diff(logits) - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE)

print(f"Clean logit diff: {CLEAN_BASELINE:.4f}")
print(f"Corrupted logit diff: {CORRUPTED_BASELINE:.4f}")
print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")

# Step 2: Trace and access .value safely
clean_vals, corr_vals, grad_vals = [], [], []

with model.trace() as tracer:
    # Clean pass
    with tracer.invoke(clean_tokens):
        for layer in model.transformer.h:
            clean_vals.append(layer.attn.c_proj.input.save())

    # Corrupted pass
    with tracer.invoke(corrupted_tokens):
        for layer in model.transformer.h:
            corr_vals.append(layer.attn.c_proj.input.save())
            grad_vals.append(layer.attn.c_proj.input.grad.save())
        logits = model.lm_head.output.save()
        metric = ioi_metric(logits.cpu())
        metric.backward()

    # Access values ONLY after backward inside this block
    clean_vals = [val.value for val in clean_vals]
    corr_vals = [val.value for val in corr_vals]
    grad_vals = [val.value for val in grad_vals]

# Step 3: Attribution Patching over Position
patching_results = []

for grad, corr, clean in zip(grad_vals, corr_vals, clean_vals):
    residual_attr = einops.reduce(
        grad[:, -1, :] * (clean[:, -1, :] - corr[:, -1, :]),
        "batch (head dim) -> head",
        "sum",
        head=12,
        dim=64,
    )
    patching_results.append(residual_attr.detach().cpu().numpy())

# Visualization
fig = px.imshow(
    patching_results,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
    title="Attribution Patching Over Attention Heads",
    labels={"x": "Head", "y": "Layer", "color": "Norm. Logit Diff"},
)
fig.show()