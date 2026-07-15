FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3-venv \
    curl \
    git \
    git-lfs \
    ca-certificates \
    build-essential \
    pkg-config \
    ffmpeg \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    libegl1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxi6 \
    libxcursor1 \
    libxkbcommon0 \
    libsm6 \
    libxmu6 \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04 ships Python 3.10, but `lerobot` requires Python >= 3.12.
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3.12-tk \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN ln -sf /usr/local/bin/pip /usr/local/bin/pip3

# Hugging Face CLI (for `huggingface-cli login`, uploads, etc.)
RUN curl -LsSf https://hf.co/cli/install.sh | bash
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Python dependencies (consolidated; no separate requirements.txt)
RUN python -m ensurepip --upgrade && python -m pip install --upgrade pip setuptools wheel
# LeRobot 0.5.x requires torch>=2.7 and provides LeRobotDataset v3.0 (codebase_version v3.0).
RUN python -m pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu124
RUN python -m pip install --no-cache-dir \
    mujoco==3.1.6 \
    pyautogui \
    matplotlib \
    scipy \
    numpy \
    Pillow \
    glfw \
    termcolor \
    PyYAML \
    mink
RUN python -m pip install --no-cache-dir "lerobot==0.5.1"

RUN python -m pip install --no-cache-dir jupyterlab ipykernel ipywidgets

COPY . /workspace

RUN if [ -f /workspace/asset/objaverse/plate_11.zip ]; then \
    unzip -o /workspace/asset/objaverse/plate_11.zip -d /workspace/asset/objaverse/; \
    fi

ENV MPLBACKEND=module://matplotlib_inline.backend_inline \
    MUJOCO_GL=egl

CMD ["bash"]