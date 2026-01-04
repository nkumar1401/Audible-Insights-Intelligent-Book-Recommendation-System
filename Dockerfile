# Use 3.12-slim (more modern but stable)
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    python3-all-dev \
    flac \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip inside the container
RUN pip install --upgrade pip

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]