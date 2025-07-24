########################################################################### CPU ######################################################################
# Dockerfile for CPU
# This Dockerfile sets up a Python environment for running a Streamlit application.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8051

CMD ["streamlit","run","main.py","--server.port=8051","--server.address=0.0.0.0"]

########################################################################### CPU ######################################################################







# ########################################################################### GPU ######################################################################
## Dockerfile for GPU 
# Uncomment the following lines if you want to use a GPU-enabled Docker image and comment the upper lines for cpu section

# FROM python:3.10-slim

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     git \
#     ffmpeg \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# COPY . .

# RUN pip install --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt && \
#     pip install bitsandbytes==0.46.1 --no-deps

# EXPOSE 8501

# CMD ["streamlit","run","main.py","--server.port=8051","--server.address=0.0.0.0"]
# ########################################################################### GPU ######################################################################
# ########################################################################### CPU ######################################################################