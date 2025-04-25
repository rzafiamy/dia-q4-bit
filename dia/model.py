from enum import Enum

import dac
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from .audio import apply_audio_delay, build_delay_indices, build_revert_indices, decode, revert_audio_delay
from .config import DiaConfig
from .layers import DiaModel
from .state import DecoderInferenceState, DecoderOutput, EncoderInferenceState

import time
import logging

logger = logging.getLogger("Dia")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


DEFAULT_SAMPLE_RATE = 44100


def _get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int | None = None,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature
    if cfg_filter_top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask.scatter_(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV[..., 1:] = sorted_indices_to_remove_BCxV[..., :-1].clone()
        sorted_indices_to_remove_BCxV[..., 0] = 0

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV.scatter_(dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV)
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C


class ComputeDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_dtype(self) -> torch.dtype:
        if self == ComputeDtype.FLOAT32:
            return torch.float32
        elif self == ComputeDtype.FLOAT16:
            return torch.float16
        elif self == ComputeDtype.BFLOAT16:
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported compute dtype: {self}")

from bitsandbytes.nn import Linear4bit
import traceback
import torch.nn as nn

class Dia:
    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
    ):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.
            device: The device to load the model onto. If None, will automatically select the best available device.

        Raises:
            RuntimeError: If there is an error loading the DAC model.
        """
        super().__init__()
        self.config = config
        self.device = device if device is not None else _get_default_device()
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)
        self.compute_dtype = compute_dtype.to_dtype()
        self.model = DiaModel(config, self.compute_dtype)
        self.dac_model = None

    @classmethod
    def from_local(
        cls,
        config_path: str,
        checkpoint_path: str,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        quantize: bool = False,
        quantize_4bit: bool = False,
    ) -> "Dia":
        print(f"[Dia] Loading config from: {config_path}")
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        print(f"[Dia] Instantiating Dia model on CPU with dtype={compute_dtype}")
        # Force CPU at init time to avoid CUDA OOM
        dia = cls(config, compute_dtype, device=torch.device("cpu"))

        print(f"[Dia] Loading checkpoint from: {checkpoint_path}")
        try:
            # Load weights on CPU first
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            print("[Dia] Checkpoint successfully loaded.")
            dia.model.load_state_dict(state_dict)
            print("[Dia] Weights successfully loaded into model.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            print("[Dia] Exception during model load:")
            traceback.print_exc()
            raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}") from e

        dia.model.eval()

        print(f"[Dia] Quantization flags -> INT8: {quantize}, 4bit: {quantize_4bit}")
        if quantize:
            dia.quantize_model()
        if quantize_4bit:
            print("[Dia] Applying 4-bit quantization...")
            dia.quantize_model_4bit()

        # Move the quantized model to GPU only after quantization
        final_device = device if device is not None else _get_default_device()
        print(f"[Dia] Moving model to: {final_device}")
        dia.device = final_device
        dia.model.to(dia.device)

        print("[Dia] Loading DAC model...")
        dia._load_dac_model()
        print("[Dia] Model fully initialized ✅")

        return dia



    def quantize_model(self):
        """Apply dynamic INT8 quantization to encoder and decoder."""
        print("Quantizing DiaModel encoder and decoder (INT8)...")
        self.model.encoder = torch.quantization.quantize_dynamic(
            self.model.encoder, {nn.Linear}, dtype=torch.qint8
        )
        self.model.decoder = torch.quantization.quantize_dynamic(
            self.model.decoder, {nn.Linear}, dtype=torch.qint8
        )

    def quantize_model_4bit(self):
        """Replace nn.Linear layers with properly initialized bnb.nn.Linear4bit layers."""

        def replace_linear(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    print(f"Converting {name} to Linear4bit")

                    new_linear = bnb.nn.Linear4bit(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.float16,
                        quant_type="nf4",  # or "fp4", "int4" if you want to test
                        compress_statistics=True,
                    )

                    # Load the pretrained FP32 weights into the 4-bit wrapper
                    new_linear.weight = nn.Parameter(child.weight.detach().clone())
                    if child.bias is not None:
                        new_linear.bias = nn.Parameter(child.bias.detach().clone())

                    setattr(module, name, new_linear)

                else:
                    replace_linear(child)

        print("Quantizing DiaModel encoder and decoder (4-bit, manual conversion)...")
        replace_linear(self.model.encoder)
        replace_linear(self.model.decoder)

        
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nari-labs/Dia-1.6B",
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        quantize: bool = False,         # INT8
        quantize_4bit: bool = False,    # INT4 (bnb)
    ) -> "Dia":
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        checkpoint_path = hf_hub_download(repo_id=model_name, filename="dia-v0_1.pth")
        print(f"[Dia] quantize={quantize}, quantize_4bit={quantize_4bit}")
        return cls.from_local(
            config_path,
            checkpoint_path,
            compute_dtype,
            device,
            quantize=quantize,
            quantize_4bit=quantize_4bit,  # << pass it through
        )



    def _load_dac_model(self):
        try:
            dac_model_path = dac.utils.download()
            dac_model = dac.DAC.load(dac_model_path).to(self.device)
        except Exception as e:
            raise RuntimeError("Failed to load DAC model") from e
        self.dac_model = dac_model

    def _prepare_text_input(self, text: str) -> torch.Tensor:
        """Encodes text prompt, pads, and creates attention mask and positions."""
        text_pad_value = self.config.data.text_pad_value
        max_len = self.config.data.text_length

        byte_text = text.encode("utf-8")
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)

        current_len = len(text_tokens)
        padding_needed = max_len - current_len
        if padding_needed <= 0:
            text_tokens = text_tokens[:max_len]
            padded_text_np = np.array(text_tokens, dtype=np.uint8)
        else:
            padded_text_np = np.pad(
                text_tokens,
                (0, padding_needed),
                mode="constant",
                constant_values=text_pad_value,
            ).astype(np.uint8)

        src_tokens = torch.from_numpy(padded_text_np).to(torch.long).to(self.device).unsqueeze(0)  # [1, S]
        return src_tokens

    def _prepare_audio_prompt(self, audio_prompt: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        num_channels = self.config.data.channels
        audio_bos_value = self.config.data.audio_bos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_delay_pattern = max(delay_pattern)

        prefill = torch.full(
            (1, num_channels),
            fill_value=audio_bos_value,
            dtype=torch.int,
            device=self.device,
        )

        prefill_step = 1

        if audio_prompt is not None:
            prefill_step += audio_prompt.shape[0]
            prefill = torch.cat([prefill, audio_prompt], dim=0)

        delay_pad_tensor = torch.full(
            (max_delay_pattern, num_channels), fill_value=-1, dtype=torch.int, device=self.device
        )
        prefill = torch.cat([prefill, delay_pad_tensor], dim=0)

        delay_precomp = build_delay_indices(
            B=1,
            T=prefill.shape[0],
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        prefill = apply_audio_delay(
            audio_BxTxC=prefill.unsqueeze(0),
            pad_value=audio_pad_value,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        ).squeeze(0)

        return prefill, prefill_step

    def _prepare_generation(self, text: str, audio_prompt: str | torch.Tensor | None, verbose: bool):
        enc_input_cond = self._prepare_text_input(text)
        enc_input_uncond = torch.zeros_like(enc_input_cond)
        enc_input = torch.cat([enc_input_uncond, enc_input_cond], dim=0)

        if isinstance(audio_prompt, str):
            audio_prompt = self.load_audio(audio_prompt)
        prefill, prefill_step = self._prepare_audio_prompt(audio_prompt)

        if verbose:
            print("generate: data loaded")

        enc_state = EncoderInferenceState.new(self.config, enc_input_cond)
        encoder_out = self.model.encoder(enc_input, enc_state)

        dec_cross_attn_cache = self.model.decoder.precompute_cross_attn_cache(encoder_out, enc_state.positions)
        dec_state = DecoderInferenceState.new(
            self.config, enc_state, encoder_out, dec_cross_attn_cache, self.compute_dtype
        )
        dec_output = DecoderOutput.new(self.config, self.device)
        dec_output.prefill(prefill, prefill_step)

        dec_step = prefill_step - 1
        if dec_step > 0:
            dec_state.prepare_step(0, dec_step)
            tokens_BxTxC = dec_output.get_tokens_at(0, dec_step).unsqueeze(0).expand(2, -1, -1)
            self.model.decoder.forward(tokens_BxTxC, dec_state)

        return dec_state, dec_output

    def _decoder_step(
        self,
        tokens_Bx1xC: torch.Tensor,
        dec_state: DecoderInferenceState,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        cfg_filter_top_k: int,
    ) -> torch.Tensor:
        audio_eos_value = self.config.data.audio_eos_value
        logits_Bx1xCxV = self.model.decoder.decode_step(tokens_Bx1xC, dec_state)

        logits_last_BxCxV = logits_Bx1xCxV[:, -1, :, :]
        uncond_logits_CxV = logits_last_BxCxV[0, :, :]
        cond_logits_CxV = logits_last_BxCxV[1, :, :]

        logits_CxV = cond_logits_CxV + cfg_scale * (cond_logits_CxV - uncond_logits_CxV)
        logits_CxV[:, audio_eos_value + 1 :] = -torch.inf
        logits_CxV[1:, audio_eos_value:] = -torch.inf

        pred_C = _sample_next_token(
            logits_CxV.float(),
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
        )
        return pred_C

    def _generate_output(self, generated_codes: torch.Tensor) -> np.ndarray:
        num_channels = self.config.data.channels
        seq_length = generated_codes.shape[0]
        delay_pattern = self.config.data.delay_pattern
        audio_pad_value = self.config.data.audio_pad_value
        max_delay_pattern = max(delay_pattern)

        revert_precomp = build_revert_indices(
            B=1,
            T=seq_length,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        codebook = revert_audio_delay(
            audio_BxTxC=generated_codes.unsqueeze(0),
            pad_value=audio_pad_value,
            precomp=revert_precomp,
            T=seq_length,
        )[:, :-max_delay_pattern, :]

        min_valid_index = 0
        max_valid_index = 1023
        invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
        codebook[invalid_mask] = 0

        audio = decode(self.dac_model, codebook.transpose(1, 2))

        return audio.squeeze().cpu().numpy()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        audio, sr = torchaudio.load(audio_path, channels_first=True)  # C, T
        if sr != DEFAULT_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, DEFAULT_SAMPLE_RATE)
        audio = audio.to(self.device).unsqueeze(0)  # 1, C, T
        audio_data = self.dac_model.preprocess(audio, DEFAULT_SAMPLE_RATE)
        _, encoded_frame, _, _, _ = self.dac_model.encode(audio_data)  # 1, C, T
        return encoded_frame.squeeze(0).transpose(0, 1)

    def save_audio(self, path: str, audio: np.ndarray):
        import soundfile as sf

        sf.write(path, audio, DEFAULT_SAMPLE_RATE)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        max_tokens: int | None = None,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        use_torch_compile: bool = False,
        cfg_filter_top_k: int = 35,
        audio_prompt: str | torch.Tensor | None = None,
        audio_prompt_path: str | None = None,
        use_cfg_filter: bool | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        logger.info("Starting generation for text prompt")
        total_start = time.time()

        # Fast handling of deprecated args
        if audio_prompt_path:
            logger.warning("audio_prompt_path is deprecated. Use audio_prompt instead.")
            audio_prompt = audio_prompt_path

        # Constants and config
        model = self.model
        model.eval()
        cfg = self.config.data
        max_tokens = cfg.audio_length if max_tokens is None else max_tokens
        max_delay_pattern = max(cfg.delay_pattern)

        # Preparation phase
        prep_start = time.time()
        dec_state, dec_output = self._prepare_generation(text, audio_prompt, verbose)
        logger.info(f"Preparation time: {time.time() - prep_start:.3f}s")

        # Decoder function
        step_fn = (
            torch.compile(self._decoder_step, mode="default", dynamic=True)
            if use_torch_compile else self._decoder_step
        )

        dec_step = dec_output.prefill_step - 1
        bos_countdown = max_delay_pattern
        eos_detected, eos_countdown = False, -1

        logger.info(f"Begin generation loop at step {dec_step}")
        log_interval = 50
        gen_start = time.time()

        # Cache tensors/constants
        delay_pattern = cfg.delay_pattern
        audio_eos_value = cfg.audio_eos_value
        audio_pad_value = cfg.audio_pad_value

        while dec_step < max_tokens:
            loop_start = time.time()

            # Prepare model for current step
            dec_state.prepare_step(dec_step)

            tokens = dec_output.get_tokens_at(dec_step).unsqueeze(0).expand(2, -1, -1)

            # Decoder step
            pred = step_fn(tokens, dec_state, cfg_scale, temperature, top_p, cfg_filter_top_k)

            # EOS logic (manual scheduling)
            if (not eos_detected and pred[0] == audio_eos_value) or dec_step == max_tokens - max_delay_pattern - 1:
                eos_detected = True
                eos_countdown = max_delay_pattern

            if eos_countdown > 0:
                step_after_eos = max_delay_pattern - eos_countdown
                for i, d in enumerate(delay_pattern):
                    if step_after_eos == d:
                        pred[i] = audio_eos_value
                    elif step_after_eos > d:
                        pred[i] = audio_pad_value
                eos_countdown -= 1

            # Update
            bos_countdown = max(0, bos_countdown - 1)
            dec_output.update_one(pred, dec_step + 1, bos_countdown > 0)
            dec_step += 1

            if dec_step % log_interval == 0:
                elapsed = time.time() - loop_start
                logger.info(
                    f"Step {dec_step}: {log_interval / elapsed:.2f} tok/s; "
                    f"total gen={time.time() - gen_start:.3f}s"
                )

            if eos_countdown == 0:
                logger.info("EOS completed, exiting generation loop")
                break

        logger.info(f"Total generation time: {time.time() - gen_start:.3f}s over {dec_step} steps")

        # Final decoding to audio
        decode_start = time.time()
        codes = dec_output.generated_tokens[dec_output.prefill_step : dec_step + 1, :]
        audio = self._generate_output(codes)
        logger.info(f"Decoding to audio took: {time.time() - decode_start:.3f}s")

        logger.info(f"End-to-end generation time: {time.time() - total_start:.3f}s")
        return audio


