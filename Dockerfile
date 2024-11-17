# Use a base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y curl libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and source code
COPY requirements.txt /app/requirements.txt
COPY src /app/src

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Set the entry point
ENTRYPOINT ["bash"]
