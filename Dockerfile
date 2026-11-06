# CPU-default image; build with docker build (BuildKit).
#
# Architecture: on linux/arm64 the meds-reader and xformers sdists have no
# wheels, so the builder stage installs bazelisk (checksum-pinned) and
# builds both from source during the locked sync. On linux/amd64 the
# manylinux wheels are used and the build is fast.
#
# CUDA: the default build installs the locked CPU torch. Rebuild with
#   --build-arg TORCH_VARIANT=cu126
# to swap in the CUDA build of the same torch version afterwards.

FROM python:3.12-slim AS builder

ARG TARGETARCH
ARG TORCH_VARIANT=cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /usr/local/bin/

ENV UV_PYTHON=/usr/local/bin/python3 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# A builder home keeps the uv and bazel caches on writable, non-absolute-
# user paths (cache mounts below).
RUN useradd --create-home builder

# bazelisk drives meds-reader's sdist build on arm64 (its setup.py probes
# with a small bazel build and falls back to an arch-flag-free bazelrc).
# The release binary is checksum-pinned; a `python` symlink is needed by
# bazel's rules_python.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      curl -fsSL -o /usr/local/bin/bazelisk \
        "https://github.com/bazelbuild/bazelisk/releases/download/v1.30.0/bazelisk-linux-arm64" \
      && echo "b2bcc040c213b44586b121bda6d983c35df121b917c3a7a66fb16f2fa0df1a4f  /usr/local/bin/bazelisk" \
        | sha256sum -c - \
      && chmod +x /usr/local/bin/bazelisk \
      && ln -sf /usr/local/bin/bazelisk /usr/local/bin/bazel \
      && ln -sf /usr/local/bin/python3 /usr/local/bin/python; \
    fi

USER builder
WORKDIR /home/builder/app

# The cache mounts below need builder-owned parent directories (mount
# targets are otherwise created root-owned and bazel's mkdir fails).
RUN mkdir -p /home/builder/.cache/bazel /home/builder/.cache/bazelisk

COPY --chown=builder:builder pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/home/builder/.cache/uv,uid=1000 \
    --mount=type=cache,target=/home/builder/.cache/bazelisk,uid=1000 \
    uv sync --frozen --no-dev --no-install-project

COPY --chown=builder:builder src ./src
COPY --chown=builder:builder LICENSE LICENSES/ NOTICE ./

RUN --mount=type=cache,target=/home/builder/.cache/uv,uid=1000 \
    uv sync --frozen --no-dev --no-editable \
    && if [ "$TORCH_VARIANT" != "cpu" ]; then \
      uv pip install --reinstall-package torch \
        --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" torch; \
    fi

FROM python:3.12-slim

RUN groupadd --system ehrfm \
    && useradd --system --gid ehrfm --home-dir /home/ehrfm --create-home ehrfm

WORKDIR /app

COPY --from=builder --chown=ehrfm:ehrfm /home/builder/app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV=/app/.venv \
    XDG_CACHE_HOME=/home/ehrfm/.cache \
    EHRFM_SERVER_HOST=0.0.0.0 \
    EHRFM_SERVER_PORT=8000

RUN mkdir -p /home/ehrfm/.cache && chown -R ehrfm:ehrfm /home/ehrfm

USER ehrfm

EXPOSE 8000

CMD ["ehrfm", "serve", "--toy"]
