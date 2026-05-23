"""
Orpheus TTS — RunPod Serverless Handler
Supports built-in voices and zero-shot voice cloning (base64 WAV per-request).
Models are downloaded on first startup and cached on RunPod container disk.
"""

import os
import glob

# ─── Fix CUDA linker paths BEFORE importing torch/vllm ───────────────────────
def _fix_cuda_links():
    candidates = glob.glob("/usr/local/cuda-*/compat/libcuda.so.*")
    if not candidates:
        candidates = glob.glob("/usr/local/cuda/lib64/stubs/libcuda.so*")
    if not candidates:
        print("[CUDA Fix] WARNING: Could not locate libcuda.so — may be fine on RunPod.")
        return
    source = candidates[0]
    targets = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib/libcuda.so",
        "/usr/lib64/libcuda.so",
    ]
    for target in targets:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.islink(target) or os.path.exists(target):
            try:
                os.remove(target)
            except Exception:
                pass
        try:
            os.symlink(source, target)
        except Exception:
            pass
    print(f"[CUDA Fix] Symlinked {source} into linker search paths.")

_fix_cuda_links()

# ─── Standard imports ─────────────────────────────────────────────────────────
import io
import re
import uuid
import base64
import asyncio

import torch
import numpy as np
import soundfile as sf
import librosa
from snac import SNAC
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import runpod

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_ID   = os.environ.get("MODEL_ID",    "heydryft/Orpheus-3b-FT-AWQ")
SNAC_ID    = os.environ.get("SNAC_ID",     "hubertsiuzdak/snac_24khz")
MODEL_PATH = os.environ.get("MODEL_PATH",  "/models/orpheus-3b-ft-awq")
SNAC_PATH  = os.environ.get("SNAC_PATH",   "/models/snac_24khz")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Token positional offsets for SNAC 7-token frame layout
POS_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]

# Built-in voice IDs supported by heydryft/Orpheus-3b-FT-AWQ
BUILTIN_VOICES = {"tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"}


# ─── Model download (runs once on first startup, cached on container disk) ────
def _ensure_models():
    """Download models if not already present on the container disk."""
    from huggingface_hub import snapshot_download

    if not os.path.isdir(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"[Startup] Downloading Orpheus model → {MODEL_PATH} ...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        snapshot_download(
            MODEL_ID,
            local_dir=MODEL_PATH,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        print("[Startup] Orpheus model downloaded.")
    else:
        print(f"[Startup] Orpheus model already cached at {MODEL_PATH}. Skipping download.")

    if not os.path.isdir(SNAC_PATH) or not os.listdir(SNAC_PATH):
        print(f"[Startup] Downloading SNAC model → {SNAC_PATH} ...")
        os.makedirs(SNAC_PATH, exist_ok=True)
        snapshot_download(SNAC_ID, local_dir=SNAC_PATH)
        print("[Startup] SNAC model downloaded.")
    else:
        print(f"[Startup] SNAC model already cached at {SNAC_PATH}. Skipping download.")


_ensure_models()

# ─── Model loading (runs once at worker startup) ──────────────────────────────
print(f"[Startup] Loading SNAC model from: {SNAC_PATH}")
snac_model = SNAC.from_pretrained(SNAC_PATH).eval().to(DEVICE)
print("[Startup] SNAC model loaded.")

print(f"[Startup] Loading vLLM engine from: {MODEL_PATH}")
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    quantization="awq",
    dtype="float16",
    tensor_parallel_size=1,
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.80,
    enforce_eager=True,
    disable_custom_all_reduce=True,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("[Startup] vLLM engine loaded. Worker is ready.")


# ─── Voice encoding (SNAC → token IDs) ───────────────────────────────────────
def encode_voice(audio_bytes: bytes) -> list:
    """
    Encode a WAV audio clip into a list of vLLM prompt token IDs
    that represent the speaker's acoustic profile for zero-shot cloning.
    """
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=24000)
    audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        codes = snac_model.encode(audio_tensor)

    c0 = codes[0].squeeze(0).cpu().numpy()
    c1 = codes[1].squeeze(0).cpu().numpy()
    c2 = codes[2].squeeze(0).cpu().numpy()

    N = len(c0)
    raw_codes = []
    for i in range(N):
        frame = [
            c0[i],
            c1[2 * i],     c2[4 * i],     c2[4 * i + 1],
            c1[2 * i + 1], c2[4 * i + 2], c2[4 * i + 3],
        ]
        for pos, sc in enumerate(frame):
            raw_codes.append(int(sc) + POS_OFFSETS[pos] + 10)

    tokens = [f"<custom_token_{c}>" for c in raw_codes]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    for i, t_id in enumerate(ids):
        if t_id is None:
            ids[i] = tokenizer.encode(tokens[i], add_special_tokens=False)[0]
    return ids


# ─── Frame decoding (SNAC tokens → waveform) ─────────────────────────────────
def decode_frames(frames: list):
    if not frames:
        return None
    c0 = torch.tensor([f[0] for f in frames],                               dtype=torch.long).unsqueeze(0)
    c1 = torch.tensor([v for f in frames for v in [f[1], f[4]]],           dtype=torch.long).unsqueeze(0)
    c2 = torch.tensor([v for f in frames for v in [f[2], f[3], f[5], f[6]]], dtype=torch.long).unsqueeze(0)

    with torch.inference_mode():
        audio = snac_model.decode([c0.to(DEVICE), c1.to(DEVICE), c2.to(DEVICE)])
    return audio.squeeze().cpu().numpy()


# ─── Core inference ───────────────────────────────────────────────────────────
async def generate_audio(
    text: str,
    voice_id: str,
    voice_token_ids: list = None,
) -> bytes:
    """Run vLLM inference and return WAV file bytes."""
    # Build the prompt using Orpheus canonical conversation format:
    #   128259 = start_of_human
    #   128000 = bos
    #   (optional voice cloning tokens)
    #   text tokens = "{voice_id}: {text}"
    #   128009 = end_of_text
    #   128260 = end_of_human
    #   128261 = start_of_ai
    #   128257 = start_of_speech (audio trigger)
    # Getting these in the wrong order or missing tokens causes garbage output.
    prompt_ids = [128259, 128000]

    if voice_token_ids:
        prompt_ids.extend(voice_token_ids)

    text_tokens = tokenizer.encode(f"{voice_id}: {text}", add_special_tokens=False)
    prompt_ids.extend(text_tokens)
    prompt_ids.extend([128009, 128260, 128261, 128257])

    # Resolve the end-of-audio stop token ID from the tokenizer.
    # <custom_token_2> is the Orpheus end-of-audio signal.
    # We try convert_tokens_to_ids first; if it returns None (happens with some AWQ
    # checkpoints), we fall back to encoding the string. We log the resolved ID so
    # we can verify it is correct from the RunPod logs.
    stop_id = tokenizer.convert_tokens_to_ids("<custom_token_2>")
    if stop_id is None or stop_id == tokenizer.unk_token_id:
        encoded = tokenizer.encode("<custom_token_2>", add_special_tokens=False)
        stop_id = encoded[0] if encoded else 128261
    print(f"[Handler] Stop token ID resolved: {stop_id}")

    # Increased max_tokens to allow longer generations (up to ~30 seconds).
    # The loop detector handles any overshoot or hallucinations.
    max_tokens = 2048
    print(f"[Handler] max_tokens={max_tokens}")

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.3,
        presence_penalty=0.5,
        max_tokens=max_tokens,
        stop_token_ids=[stop_id, 128001, 128009],
    )

    request_id = str(uuid.uuid4())
    results_generator = engine.generate(
        {"prompt_token_ids": prompt_ids},
        sampling_params,
        request_id,
    )

    # Streaming real-time loop detection.
    # NOTE: vLLM accumulates the full text on each iteration.
    # We track how many complete 7-token audio frames we've already processed
    # using `frames_processed` (frame count, not token count) to avoid
    # reprocessing the same frames on each streaming step.
    frames_processed = 0
    recent_frames = []
    frames = []
    loop_detected = False

    async for request_output in results_generator:
        new_text = request_output.outputs[0].text

        # Parse ALL audio token numbers from the accumulated text so far
        name_nums = [int(m) for m in re.findall(r"<custom_token_(\d+)>", new_text)]
        audio_nums = [n for n in name_nums if n >= 10]
        n_full_frames = len(audio_nums) // 7

        # Only process newly available frames (beyond what we've seen already)
        if n_full_frames > frames_processed:
            for fi in range(frames_processed, n_full_frames):
                i = fi * 7
                frame = []
                valid = True
                for pos in range(7):
                    raw_code = audio_nums[i + pos] - 10
                    snac_code = raw_code - POS_OFFSETS[pos]
                    if not (0 <= snac_code < 4096):
                        valid = False
                        break
                    frame.append(snac_code)

                if valid:
                    frames.append(frame)
                    recent_frames.append(frame)
                    if len(recent_frames) > 20:
                        recent_frames.pop(0)

                    # Real-time loop detection: 4 identical frames in a row = hallucination
                    if len(recent_frames) >= 8 and recent_frames[-4:] == recent_frames[-8:-4]:
                        print("[Handler] Loop detected — aborting engine.")
                        await engine.abort(request_id)
                        loop_detected = True
                        break

            frames_processed = n_full_frames

        if loop_detected:
            frames = frames[:-4]  # strip the repeated tail
            break

    if not frames:
        raise RuntimeError("No valid audio frames decoded from model output.")

    wav = decode_frames(frames)
    if wav is None:
        raise RuntimeError("SNAC audio decoding returned None.")

    # Encode to WAV bytes
    buf = io.BytesIO()
    sf.write(buf, wav, 24000, format="WAV")
    buf.seek(0)
    return buf.read()


# ─── RunPod handler ───────────────────────────────────────────────────────────
async def handler(job):
    """
    RunPod job handler.

    Expected input:
    {
        "action": "tts",                  # required
        "text": "Hello world",            # required
        "voice_id": "tara",               # required — built-in or custom name
        "voice_audio_b64": "<base64>"     # optional — base64-encoded WAV for cloning
    }

    Built-in voice IDs: tara, leah, jess, leo, dan, mia, zac, zoe
    """
    job_input = job.get("input", {})

    # ── Validate inputs ────────────────────────────────────────────────────────
    action = job_input.get("action", "tts")
    if action != "tts":
        return {"error": f"Unknown action '{action}'. Supported actions: 'tts'"}

    text = job_input.get("text", "").strip()
    if not text:
        return {"error": "Missing required field: 'text'"}

    voice_id        = job_input.get("voice_id", "tara")
    voice_audio_b64 = job_input.get("voice_audio_b64", None)

    # ── Resolve voice tokens ───────────────────────────────────────────────────
    voice_token_ids = None

    if voice_audio_b64:
        # Zero-shot voice cloning: decode base64 WAV → SNAC tokens
        try:
            audio_bytes = base64.b64decode(voice_audio_b64)
            voice_token_ids = encode_voice(audio_bytes)
            print(f"[Handler] Voice '{voice_id}' encoded: {len(voice_token_ids)} tokens")
        except Exception as e:
            return {"error": f"Failed to encode voice audio: {str(e)}"}
    elif voice_id not in BUILTIN_VOICES:
        return {
            "error": (
                f"Unknown voice_id '{voice_id}'. "
                f"Built-in voices: {sorted(BUILTIN_VOICES)}. "
                "For custom voice cloning, also provide 'voice_audio_b64' (base64 WAV)."
            )
        }

    # ── Generate audio ─────────────────────────────────────────────────────────
    print(f"[Handler] Generating TTS: voice='{voice_id}', text='{text[:60]}...'")
    try:
        wav_bytes = await generate_audio(text, voice_id, voice_token_ids)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        print(f"[Handler] Done. WAV size: {len(wav_bytes):,} bytes")
        return {
            "audio_b64":   audio_b64,
            "sample_rate": 24000,
            "voice_id":    voice_id,
            "format":      "wav",
        }
    except Exception as e:
        print(f"[Handler] ERROR: {e}")
        return {"error": str(e)}


# ─── Start RunPod serverless worker ───────────────────────────────────────────
runpod.serverless.start({"handler": handler})