FROM python:3.9-slim
WORKDIR /app
RUN apt-get update \
    && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1 libice6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    torch torchvision \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN git clone -b LiteMedSAM https://github.com/bowang-lab/MedSAM.git
WORKDIR /app/MedSAM
RUN pip install -e .

WORKDIR /app/python
COPY python/ .
WORKDIR /app/python/pretrained_segmenters/MedSAM
RUN wget -O lite_medsam.pth https://github.com/uw0s/DICOMDeIdentifier/releases/download/v1.3.5/lite_medsam.pth

WORKDIR /app/python
COPY python/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
