FROM python:3.9-slim
RUN apt-get update && apt-get install git libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1 libice6 -y
WORKDIR /usr/src/app
COPY pyproject.toml .
RUN python3 -m pip install --upgrade pip && python3 -m pip install -e .[dev]
RUN python3 -m pip install -e git+https://github.com/bowang-lab/MedSAM.git@LiteMedSAM#egg=litemedsam
