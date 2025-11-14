# Multi-stage Dockerfile for VibeVoice
# This Dockerfile is completely self-contained and requires no local source code
# All source code is cloned from GitHub during build

ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ARG WORKDIR=/workspace/zhao-kun/vibevoice
ARG GITHUB_REPO=https://github.com/zhao-kun/vibevoice.git
ARG GITHUB_BRANCH=main

#############################################
# Stage 1: Download model from HuggingFace
#############################################
FROM python:3.10-slim AS model-downloader

# Install huggingface-cli
RUN pip install --no-cache-dir huggingface-hub[cli]

# Set working directory
WORKDIR /tmp/models

# Download model (float8_e4m3fn only) using huggingface-cli
RUN hf download zhaokun/vibevoice-large \
    vibevoice7b_float8_e4m3fn.safetensors \
    --local-dir /tmp/models/VibeVoice-large

RUN hf download zhaokun/vibevoice-large \
    vibevoice7b_bf16.safetensors \
    --local-dir /tmp/models/VibeVoice-large

#############################################
# Stage 2: Clone Repository and Build Frontend
#############################################
FROM node:20-alpine AS source-and-frontend

ARG GITHUB_REPO
ARG GITHUB_BRANCH

# Install git
RUN apk add --no-cache git

# Clone repository
WORKDIR /build
RUN git clone --depth 1 --branch ${GITHUB_BRANCH} ${GITHUB_REPO} vibevoice && \
    cd /build/vibevoice && \
    git checkout main && \
    git rev-parse HEAD > backend/version.txt

# Build frontend
WORKDIR /build/vibevoice/frontend

# Install dependencies
RUN npm ci

# Build frontend
RUN npm run build

# Verify build output
RUN ls -la out/

#############################################
# Stage 3: Create Python Virtual Environment
#############################################
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS python-builder

ARG WORKDIR=/workspace/zhao-kun/vibevoice
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --allow-releaseinfo-change --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends \
    bash \
    libgl1 \
    software-properties-common \
    ffmpeg \
    zip \
    unzip \
    iputils-ping \
    libtcmalloc-minimal4 \
    net-tools \
    vim \
    p7zip-full && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update --allow-releaseinfo-change --yes && \
    apt install python3.10-dev python3.10-venv python3-pip \
    build-essential git curl -y --no-install-recommends && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Create working directory at EXACT runtime path
RUN mkdir -p ${WORKDIR}
WORKDIR ${WORKDIR}

# Copy source code from frontend stage
COPY --from=source-and-frontend /build/vibevoice .

# Create virtual environment at runtime path (critical for absolute paths in venv)
RUN python3.10 -m venv ${WORKDIR}/venv

# Upgrade pip and install dependencies
RUN ${WORKDIR}/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    ${WORKDIR}/venv/bin/pip install --no-cache-dir .

RUN rm -rf ${WORKDIR}/frontend

#############################################
# Stage 4: Final Image
#############################################
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ARG WORKDIR=/workspace/zhao-kun/vibevoice
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update --allow-releaseinfo-change --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends \
    bash \
    libgl1 \
    software-properties-common \
    ffmpeg \
    zip \
    unzip \
    iputils-ping \
    libtcmalloc-minimal4 \
    net-tools \
    vim \
    p7zip-full && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update --allow-releaseinfo-change --yes && \
    apt install python3.10-dev python3.10-venv python3-pip \
    build-essential git curl -y --no-install-recommends && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen


# Copy downloaded model from model-downloader stage
RUN mkdir -p /tmp/models/
COPY --from=model-downloader /tmp/models/VibeVoice-large /tmp/models

# Create working directory at EXACT same path as build stage
RUN mkdir -p ${WORKDIR}
WORKDIR ${WORKDIR}
# Copy virtual environment from python-builder stage (with preserved absolute paths)
COPY --from=python-builder ${WORKDIR} .

RUN mkdir -p ${WORKDIR}/models/vibevoice/  && ln -s /tmp/models/vibevoice7b_float8_e4m3fn.safetensors ${WORKDIR}/models/vibevoice/
RUN mkdir -p ${WORKDIR}/models/vibevoice/  && ln -s /tmp/models/vibevoice7b_bf16.safetensors ${WORKDIR}/models/vibevoice/

# Copy frontend build from source-and-frontend stage
RUN mkdir -p ${WORKDIR}/backend/dist
COPY --from=source-and-frontend /build/vibevoice/frontend/out ${WORKDIR}/backend/dist

# Create workspace directory for runtime data
RUN mkdir -p ${WORKDIR}/workspace

# Expose port
EXPOSE 9527

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9527/health || exit 1

# Use venv python explicitly (critical - do not rely on PATH)
CMD ["/workspace/zhao-kun/vibevoice/venv/bin/python", "backend/run.py"]

