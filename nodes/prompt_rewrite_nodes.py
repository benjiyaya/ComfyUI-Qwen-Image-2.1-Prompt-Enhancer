"""Qwen-Image-2.1 Prompt Enhancer — ComfyUI Custom Node.

Uses CLIPLoader to load the PE checkpoint as a text encoder, then runs
autoregressive generation via clip.tokenize() + clip.generate() + clip.decode()
— the same native path as ComfyUI's built-in TextGenerate node.

User workflow:
  1. Drop PE checkpoint .safetensors into models/text_encoders/
  2. Add CLIPLoader node, pick the PE file from dropdown
  3. Connect CLIP output to this node
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("comfyui-qwen-prompt-rewrite")

_VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def _chat_prompt(system_prompt: str, user_text: str) -> str:
    # Qwen chat format, assembled manually (not via llama_template) because the
    # PE system prompts contain literal braces that the template .format() would eat.
    return (
        "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
        "<|im_start|>user\n" + user_text + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# System prompt loading
# ---------------------------------------------------------------------------
def _load_system_prompt(task: str) -> str:
    from pathlib import Path

    # Prompt files live at the repository root, alongside the nodes directory.
    path = Path(__file__).resolve().parent.parent / "prompts" / f"system_prompt_{task}.txt"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Unable to read system prompt: {path}") from exc
    if not text:
        raise RuntimeError(f"Empty system prompt: {path}")
    return text


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------
def _balanced_braces(text: str) -> list[str]:
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : i + 1])
    return spans


def _parse_json_answer(answer: str, has_ratio_follow: bool = False) -> dict[str, Any]:
    answer = (answer or "").strip()
    for candidate in reversed(_balanced_braces(answer)):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                import json_repair
                obj = json_repair.repair_json(candidate, return_objects=True)
                if isinstance(obj, list):
                    obj = obj[0] if obj else None
            except Exception:
                continue
        if not isinstance(obj, dict):
            continue
        rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
        if isinstance(rewritten, str) and rewritten.strip():
            return {
                "positive_prompt": rewritten.strip(),
                "wh_ratio": str(obj.get("wh_ratio") or "").strip(),
                "ratio_follow": (
                    str(obj.get("ratio_follow") or "").strip()
                    if has_ratio_follow else ""
                ),
                "parse_ok": True,
            }
    return {"positive_prompt": answer, "wh_ratio": "", "ratio_follow": "", "parse_ok": False}


def _split_thinking(text: str) -> tuple[str, str]:
    if "</think>" in text:
        think, _, answer = text.partition("</think>")
        if "<think>" in think:
            think = think.partition("<think>")[2]
        return think.strip(), answer.strip()
    if "<think>" in text:
        return text.partition("<think>")[2].strip(), ""
    return "", text.strip()


# ===========================================================================
# T2I Prompt Rewrite Node
# ===========================================================================
class QwenImage21_T2IPromptRewrite:
    """Expand a short text prompt into a detailed image description."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Short text prompt (any language)",
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "presence_penalty": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Should be 1.5 for T2I",
                }),
                "max_new_tokens": ("INT", {
                    "default": 16256, "min": 256, "max": 32768, "step": 256,
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "wh_ratio", "thinking", "parse_ok")
    FUNCTION = "rewrite"
    CATEGORY = "Qwen Image"
    DESCRIPTION = "Expand a short text prompt into a detailed image description."
    OUTPUT_NODE = False

    def rewrite(
        self,
        clip,
        prompt,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_new_tokens=16256,
        seed=42,
    ):
        if not prompt.strip():
            return ("", "", "", "", True)

        system_prompt = _load_system_prompt("t2i")

        tokens = clip.tokenize(
            _chat_prompt(system_prompt, prompt),
            skip_template=True,
            min_length=1,
            thinking=True,
        )
        generated_ids = clip.generate(
            tokens,
            do_sample=True,
            max_length=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=presence_penalty,
            seed=seed,
        )
        raw_text = clip.decode(generated_ids)
        thinking_text, answer_text = _split_thinking(raw_text)
        result = _parse_json_answer(answer_text, has_ratio_follow=False)

        if not result["parse_ok"]:
            logger.warning("T2I: answer did not parse as JSON, using raw text")

        return (
            result["positive_prompt"],
            "",  # negative_prompt always empty
            result["wh_ratio"],
            thinking_text,
            result["parse_ok"],
        )


# ===========================================================================
# Edit Prompt Rewrite Node
# ===========================================================================
_MAX_IMAGES = 10


class QwenImage21_EditPromptRewrite:
    """Rewrite an edit instruction with 1-10 source images into a precise prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        images = {}
        for i in range(1, _MAX_IMAGES + 1):
            images["image_%d" % i] = ("IMAGE", {
                "optional": True,
                "tooltip": "Reference image #%d — use <image%d> in prompt" % (i, i),
            })
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Edit instruction — use <image1>, <image2> etc.",
                }),
            },
            "optional": {
                **images,
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Should be 0 for Edit",
                }),
                "max_length": ("INT", {
                    "default": 24000, "min": 256, "max": 32768, "step": 256,
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "wh_ratio", "ratio_follow", "thinking", "parse_ok")
    FUNCTION = "rewrite"
    CATEGORY = "Qwen Image"
    DESCRIPTION = "Rewrite an edit instruction with 1-10 source images into a precise prompt."
    OUTPUT_NODE = False

    def rewrite(
        self,
        clip,
        prompt,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=0.0,
        max_length=24000,
        seed=42,
        **kwargs,
    ):
        if not prompt.strip():
            return ("", "", "", "", "", True)

        system_prompt = _load_system_prompt("edit")

        # Collect images in order, splitting batch dims into individual images
        image_list = []
        for i in range(1, _MAX_IMAGES + 1):
            key = "image_%d" % i
            tensor = kwargs.get(key)
            if tensor is None:
                continue
            for b in range(tensor.shape[0]):
                image_list.append(tensor[b : b + 1])

        if not image_list:
            logger.warning("Edit node: no images connected")
            return ("", "", "", "", "", True)

        logger.info("Edit prompt rewrite: %d image(s)" % len(image_list))

        # <image1> <image2> ... before the text, matching the native Qwen-Image 2.1
        # tokenizer which splices each image into its vision block in order.
        refs = " ".join("<image%d> %s" % (i + 1, _VISION_BLOCK) for i in range(len(image_list)))
        user_text = refs + " " + prompt

        tokens = clip.tokenize(
            _chat_prompt(system_prompt, user_text),
            images=image_list,
            skip_template=True,
            min_length=1,
            thinking=True,
        )
        generated_ids = clip.generate(
            tokens,
            do_sample=True,
            max_length=max_length,
            temperature=temperature,
            top_k=20,
            top_p=top_p,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=presence_penalty,
            seed=seed,
        )
        raw_text = clip.decode(generated_ids)
        thinking_text, answer_text = _split_thinking(raw_text)
        result = _parse_json_answer(answer_text, has_ratio_follow=True)

        if not result["parse_ok"]:
            logger.warning("Edit: answer did not parse as JSON, using raw text")

        return (
            result["positive_prompt"],
            "",
            result["wh_ratio"],
            result["ratio_follow"],
            thinking_text,
            result["parse_ok"],
        )


# ===========================================================================
# Node registration
# ===========================================================================
NODE_CLASS_MAPPINGS = {
    "QwenImage21_T2IPromptRewrite": QwenImage21_T2IPromptRewrite,
    "QwenImage21_EditPromptRewrite": QwenImage21_EditPromptRewrite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImage21_T2IPromptRewrite": "Qwen Image 2.1 — T2I Prompt Rewrite",
    "QwenImage21_EditPromptRewrite": "Qwen Image 2.1 — Edit Prompt Rewrite",
}
