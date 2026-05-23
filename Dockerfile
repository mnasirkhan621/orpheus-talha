# ─────────────────────────────────────────────────────────────────────────────
# Orpheus TTS — RunPod Serverless Worker
# Base: vLLM OpenAI image with pre-installed vLLM, PyTorch, CUDA 12.x
# Strategy: Lightweight image — models are downloaded on first worker startup
#            and cached on RunPod's 30GB container disk for fast subsequent starts
# ─────────────────────────────────────────────────────────────────────────────
FROM vllm/vllm-openai:v0.5.4

USER root

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Model IDs (downloaded at runtime into /models on RunPod container disk)
ENV MODEL_ID=heydryft/Orpheus-3b-FT-AWQ
ENV SNAC_ID=hubertsiuzdak/snac_24khz
ENV MODEL_PATH=/models/orpheus-3b-ft-awq
ENV SNAC_PATH=/models/snac_24khz

# vLLM stability settings matching the working Kaggle server
ENV VLLM_USE_V1=0
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# ── System dependencies ───────────────────────────────────────────────────────
# Retry block to handle transient Ubuntu mirror glitches in GitHub Actions
RUN apt-get update && \
    (apt-get install -y --fix-missing libsndfile1 ffmpeg git || \
     (sleep 5 && apt-get update --fix-missing && apt-get install -y --fix-missing libsndfile1 ffmpeg git)) && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ── Copy handler ──────────────────────────────────────────────────────────────
COPY handler.py /handler.py

# ── Start the RunPod worker ───────────────────────────────────────────────────
# IMPORTANT: Reset ENTRYPOINT inherited from vllm/vllm-openai base image.
# Without this, Docker would pass our CMD as arguments to api_server.py.
ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
