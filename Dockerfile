FROM pennylaneai/pennylane:latest-lightning-gpu

# Install additional Python packages
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 

RUN pip install psutil

