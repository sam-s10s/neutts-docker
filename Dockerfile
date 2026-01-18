FROM python:3.13-trixie

# Install packages in a single layer
RUN apt update && apt upgrade -y && \
    apt install -y git

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Working directory
WORKDIR /


# --------------
# Neuphonic Packages
# --------------

# Dependencies
RUN apt install espeak-ng -y

# Copy neutts folder
COPY neutts /neutts

# Working directory
WORKDIR /neutts

# Install dependencies
RUN uv sync

# Env variables
ENV NEUPHONIC_MODEL=neuphonic/neutts-nano-q4-gguf
ENV NEUPOHNIC_VOICE=dave

# Load the images
RUN uv run docker_models.py


# --------------
# Cleanup
# --------------

# Cleanup build artifacts and downloaded files to reduce image size
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /neutts

# CMD runs your binary
CMD ["uv", "run", "proxy.py"]
