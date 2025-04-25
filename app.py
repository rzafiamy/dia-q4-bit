import argparse
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from dia.model import Dia


# ----------------------------
# Argument Parsing
# ----------------------------
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
args = parser.parse_args()


# ----------------------------
# Device Selection
# ----------------------------
def get_torch_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_torch_device(args.device)
print(f"[INFO] Using device: {device}")


_model = None

def get_model():
    global _model
    if _model is None:
        print("[INFO] Loading Nari model...")
        _model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B",
            compute_dtype="float16",
            quantize_4bit=True
        )
    return _model


# ----------------------------
# Inference Logic
# ----------------------------
def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
):
    if not text_input.strip():
        raise gr.Error("Text input cannot be empty.")

    output_audio = (44100, np.zeros(1, dtype=np.float32))
    prompt_path = None

    try:
        if audio_prompt_input:
            sr, audio_data = audio_prompt_input
            if audio_data is not None and audio_data.max() != 0:
                # Convert to mono
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=-1)
                audio_data = np.ascontiguousarray(audio_data.astype(np.float32))

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio_data, sr, subtype="FLOAT")
                    prompt_path = f.name
                    print(f"[DEBUG] Audio prompt saved to: {prompt_path}")

        start = time.time()
        with torch.inference_mode():
            model = get_model()
            output_audio_np = model.generate(
                text_input,
                max_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                use_torch_compile=False,
                audio_prompt=prompt_path,
            )
        print(f"[INFO] Inference completed in {time.time() - start:.2f}s")

        if output_audio_np is not None:
            sr = 44100
            speed_factor = np.clip(speed_factor, 0.1, 5.0)
            orig_len = len(output_audio_np)
            target_len = int(orig_len / speed_factor)

            if target_len > 0 and target_len != orig_len:
                resampled_audio = np.interp(
                    np.linspace(0, orig_len - 1, target_len),
                    np.arange(orig_len),
                    output_audio_np,
                )
            else:
                resampled_audio = output_audio_np

            resampled_audio = np.clip(resampled_audio, -1.0, 1.0)
            output_audio = (sr, (resampled_audio * 32767).astype(np.int16))
        else:
            raise gr.Error("Model generated empty output.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Inference failed: {e}")
    finally:
        if prompt_path:
            try:
                Path(prompt_path).unlink(missing_ok=True)
            except Exception as e:
                print(f"[WARN] Failed to delete temp file: {e}")

        # 🔥 Best VRAM cleanup fix
        import gc
        del output_audio_np  # Ensure output is released
        torch.cuda.empty_cache()
        gc.collect()


    return output_audio


# ----------------------------
# Gradio UI Setup
# ----------------------------
css = "#col-container {max-width: 90%; margin: auto;}"

# Load example text
default_text = "[S1] Dia is an open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] Wow. Amazing. (laughs) \n[S2] Try it now on Git hub or Hugging Face."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip() or "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")


# ----------------------------
# Gradio Interface
# ----------------------------
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Nari Text-to-Speech Synthesis")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text here...",
                value=default_text,
                lines=5,
            )
            audio_prompt_input = gr.Audio(
                label="Audio Prompt (Optional)",
                sources=["upload", "microphone"],
                type="numpy",
            )
            with gr.Accordion("Generation Parameters", open=False):
                model = get_model()
                max_new_tokens = gr.Slider(
                    label="Max New Tokens",
                    minimum=860,
                    maximum=3072,
                    value=model.config.data.audio_length,
                    step=50,
                )
                cfg_scale = gr.Slider("CFG Scale", 1.0, 5.0, 3.0, step=0.1)
                temperature = gr.Slider("Temperature", 1.0, 1.5, 1.3, step=0.05)
                top_p = gr.Slider("Top P", 0.8, 1.0, 0.95, step=0.01)
                cfg_filter_top_k = gr.Slider("CFG Filter Top K", 15, 50, 30, step=1)
                speed_factor_slider = gr.Slider("Speed Factor", 0.8, 1.0, 0.94, step=0.02)

            run_button = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Generated Audio", type="numpy")

    # Run inference on button click
    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
        ],
        outputs=[audio_output],
        api_name="generate_audio",
    )

    # Optional: Example demo inputs
    example_prompt_path = "./example_prompt.mp3"
    examples_list = [
        [
            "[S1] Oh fire! Oh my goodness! ... \n[S2] Everybody stay fucking calm!",
            None,
            3072,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
        ],
        [
            "[S1] Open weights text to dialogue model. ... \n[S2] This was Nari Labs.",
            example_prompt_path if Path(example_prompt_path).exists() else None,
            3072,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
        ],
    ]

    if examples_list:
        gr.Examples(
            examples=examples_list,
            inputs=[
                text_input,
                audio_prompt_input,
                max_new_tokens,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                speed_factor_slider,
            ],
            outputs=[audio_output],
            fn=run_inference,
            cache_examples=False,
            label="Examples (Click to Run)",
        )
    else:
        gr.Markdown("_(No examples configured or prompt file missing)_")


# ----------------------------
# App Launch
# ----------------------------
if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=args.share)
