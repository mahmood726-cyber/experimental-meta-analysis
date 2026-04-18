# Dockerfile for Experimental Meta-Analysis Framework
# ===================================================
# Addresses Editorial Review Priority 2.6: Reproducibility

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies with exact versions
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create results directory
RUN mkdir -p /app/results /app/reports /app/validation_results

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Verify installation
RUN python -c "import numpy, scipy; print('Dependencies OK')"
RUN python -c "from core_framework import DerSimonianLaird; print('Core OK')"

# Default command
CMD ["python", "-u", "tests.py"]

# Labels for metadata
LABEL maintainer="research@example.com"
LABEL version="1.1.0"
LABEL description="Experimental Meta-Analysis Framework with 300+ methods"
