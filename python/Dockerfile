FROM python:3.12-slim
RUN apt-get update && apt-get install libgl1-mesa-glx libglib2.0-0 -y
WORKDIR /usr/src/app
COPY pyproject.toml .
RUN python3 -m pip install --upgrade pip && python3 -m pip install .[dev]