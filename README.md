# Qwen Image 2.1 Prompt Enhancer — ComfyUI Custom Node

ComfyUI nodes for the **Qwen-Image-2.1 Prompt Enhancer** (Qwen3-VL 8B fine-tunes).

Two nodes:
- **T2I Prompt Rewrite** — expands a short text prompt into a detailed English description for text-to-image generation
- **Edit Prompt Rewrite** — rewrites a vague edit instruction + source image(s) into a precise, actionable prompt

<img width="694" height="618" alt="image" src="https://github.com/user-attachments/assets/15c1ca24-db7c-4b32-b996-e74604d55459" />

<img width="1622" height="636" alt="image" src="https://github.com/user-attachments/assets/d2b70d0a-5084-4010-8324-f22e1b79499d" />


<img width="723" height="993" alt="image" src="https://github.com/user-attachments/assets/21807f3b-a17d-42bb-9b6e-0cf5f605e56d" />

<img width="1756" height="573" alt="image" src="https://github.com/user-attachments/assets/8b571192-5790-4c3f-8eec-3d87883fd8d0" />


Repository: [github.com/benjiyaya/ComfyUI-Qwen-Image-2.1-Prompt-Enhancer](https://github.com/benjiyaya/ComfyUI-Qwen-Image-2.1-Prompt-Enhancer)

## Architecture

The prompt enhancer models are Qwen3-VL 8B fine-tunes loaded as text encoders through
ComfyUI's native **CLIPLoader** (`type=qwen_image`). Generation runs entirely on
ComfyUI's built-in `clip.tokenize()` → `clip.generate()` → `clip.decode()` path —
the same mechanism as the built-in TextGenerate node. No external server, no
transformers pipeline.

```
Load CLIP (type=qwen_image)  →  CLIP
         ↓
   System prompt + user prompt (+ reference images)
         ↓
   Autoregressive generation (thinking mode on)
         ↓
   JSON answer parsed → positive_prompt, wh_ratio, ratio_follow
```

## Setup

### 1. Download the PE checkpoints

Download the prompt-enhancer checkpoints from HuggingFace and place the
`.safetensors` files in your `ComfyUI/models/text_encoders/` directory:

- `Qwen/Qwen-Image-2.1-PE-T2I` — for text-to-image prompt expansion
- `Qwen/Qwen-Image-2.1-PE-I2I` — for image-editing instruction rewrite

### 2. Install this node

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/benjiyaya/ComfyUI-Qwen-Image-2.1-Prompt-Enhancer.git
```

### 3. Restart ComfyUI

The nodes appear under the **Qwen Image** category.

## Usage

1. Add a **Load CLIP** node, pick the PE `.safetensors` file, set **type** to `qwen_image`.
2. Connect its **CLIP** output to the rewrite node's `clip` input.
3. Wire `positive_prompt` into your CLIP Text Encode / prompting node.
4. For Edit with multiple images, reference them as `<image1>`, `<image2>`, ... in the prompt.

## Nodes

### T2I Prompt Rewrite

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| clip | CLIP | — | PE checkpoint loaded via native CLIPLoader (`type=qwen_image`) |
| prompt | STRING | — | Short text prompt (any language) |
| temperature | FLOAT | 1.0 | Sampling temperature |
| top_p | FLOAT | 0.95 | Top-p sampling |
| top_k | INT | 20 | Top-k sampling |
| presence_penalty | FLOAT | 1.5 | **Keep 1.5 for T2I** |
| max_new_tokens | INT | 16256 | Max generation length |
| seed | INT | 42 | Random seed |

| Output | Description |
|--------|-------------|
| positive_prompt | Rewritten detailed prompt |
| negative_prompt | Always empty (pipeline compatibility) |
| wh_ratio | Recommended aspect ratio (e.g. "16:9") |
| thinking | Model's reasoning trace |
| parse_ok | Whether the JSON answer parsed correctly |

### Edit Prompt Rewrite

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| clip | CLIP | — | PE checkpoint loaded via native CLIPLoader (`type=qwen_image`) |
| prompt | STRING | — | Edit instruction (use `<image1>`, `<image2>`... to reference images) |
| image_1 ... image_10 | IMAGE | — | Optional reference images (batch tensors are split) |
| temperature | FLOAT | 1.0 | Sampling temperature |
| top_p | FLOAT | 0.95 | Top-p sampling |
| presence_penalty | FLOAT | 0.0 | **Keep 0 for Edit** |
| max_length | INT | 24000 | Max generation length |
| seed | INT | 42 | Random seed |

| Output | Description |
|--------|-------------|
| positive_prompt | Rewritten precise edit instruction |
| negative_prompt | Always empty |
| wh_ratio | Canvas aspect ratio (if model chose one) |
| ratio_follow | Which input image to inherit the output ratio from |
| thinking | Model's reasoning trace |
| parse_ok | Whether the JSON answer parsed correctly |

## Workflow Example

```
Load CLIP (qwen_image) ─┬→ Qwen Image 2.1 — Edit Prompt Rewrite → CLIP Text Encode → KSampler
Load Image ─────────────┘                    ↓
                                      (positive_prompt)
```

```
Load CLIP (qwen_image) → Qwen Image 2.1 — T2I Prompt Rewrite ← "a corgi in the rain"
                                    ↓
                              (positive_prompt)
```

## Hardware Requirements

- **VRAM**: ~20GB for the PE model in bfloat16. A 24GB GPU is comfortable.
- VRAM is managed by ComfyUI's model management — the model is loaded and evicted like any other text encoder.

## Sampling Parameters

| Parameter | T2I | Edit | Why |
|-----------|-----|------|-----|
| presence_penalty | **1.5** | 0 | T2I needs a high penalty to avoid repetition; edit should flow naturally |
| max length | 16256 | 24000 | Edit responses are longer (includes `ratio_follow` logic) |
| thinking | on | on | Both models were trained with `<think>` blocks |

Optionally install [`json_repair`](https://pypi.org/project/json-repair/) to salvage
slightly malformed JSON answers (`pip install json-repair`).

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | ComfyUI registration |
| `nodes/prompt_rewrite_nodes.py` | Node definitions |
| `prompts/system_prompt_t2i.txt` | T2I system prompt |
| `prompts/system_prompt_edit.txt` | Edit system prompt |

## Credits

- [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) by Alibaba/Qwen team
- ComfyUI native integration pattern from [TextGenerate](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_textgen.py)

## License

MIT — see [LICENSE](LICENSE).
