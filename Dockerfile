# Dockerfile

FROM python:3.11-slim

# 1) Set working directory
WORKDIR /model

# 2) Copy only requirements first (for layer caching)
COPY requirements.txt /model

# 3) Install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy the rest of your code
COPY . /model

# 5) Expose the Flask port
EXPOSE 5000

# 6) Start the service
CMD ["python", "./services/model.py"]
