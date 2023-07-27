FROM python:3.8-slim-buster AS builder

RUN apt-get update && apt-get -y install git

# Update PIP & install package/requirements
RUN python -m pip install --upgrade pip

# Copy application files:
WORKDIR /app
COPY . /app

RUN pip install -e .

# Execute the machine learning pipeline:
CMD python pipeline.py
