## DGMD Computer Vision Training Pipeline

This code base provides a containerized data engineering pipeline that:

1. Downloads training images from DGMD S3 asset bucket
1. Creates a mapping between labels and class indexes
1. Balances images across different classes
1. Splits available images into training and validation sets
1. Converts training and validation sets into TFRecords for processing
1. Trains computer vision model to predict classes
1. Stores training metrics and model asset in MLflow

The pipeline framework is implemented in [Luigi](https://luigi.readthedocs.io/en/stable/running_luigi.html).

Models will be stored as Non-Prod by default and require an operator to "promote" models to Production based on performance and operational assessment.

### Pipeline Environmental Parameters

The pipeilne is heavily configurable using a set of environment variables to customize various pipeline runs:

* **DATA_DIR** - directory to store downloaded images in (default _./data_)
* **AWS_DATA_BUCKET** - bucket to retrieve training assets from (default _dgmd-s17-assets_)
* **AWS_DATA_PATH** - prefix key to use for training assets to be retrieved (default _train/downloaded-images_)
* **NUM_SHARDS** - number of TF Record shards to create (default _10_)
* **VALIDATION_PCNT** - percent of images to use for validatin (default _0.2_)
* **BATCH_SIZE** - training batch sizes (default _128_)
* **EPOCHS** - number of epochs to train (default _15_)
* **DECAY_RATE** - decay rate to apply to learning rate over traing cycles (default _0.5_)
* **LEARNING_RATE** - learning rate for model training (default _0.01_)
* **MODEL_NAME** - name to use to store model in mlflow (default _dgmd_model_)

### Example Usage

```bash
>> conda create --name pipeline python=3.8
>> conda activate pipeline
>> pip install -e .
>> python pipepline.py
```

__Note:__ depending on number of files to download and the number of epochs to train, pipeline run time may vary.

### Docker Image

Pipeline is automatically built into new Docker image on new code push to main branch and stored in GitHub Container Repository [here](https://github.com/orgs/hssrobotics23/packages?repo_name=model_training_pipeline).

Container will run until training completion or error in pipeline and then exit.  Containers can be schedule to run at specific times, always pulling latest data from S3 asset bucket and storing new models in MLflow.

An example of container configuration can be viewed in [DGMD app-deployment docker-compose.yml](https://github.com/hssrobotics23/app-deployment/blob/main/docker-compose.yml) and below:


```
  model-pipeline:
    image: ghcr.io/hssrobotics23/model_training_pipeline/dgmd_model_pipeline:latest
    container_name: dgmd_model_pipeline
    volumes:
      - secret-vol:/secrets
      - ./data:/app/data
      - ../model_training_pipeline:/app
    environment:
      - MLFLOW_TRACKING_URI=http://dgmd_mlflow:5000/
      - MODEL_NAME=dgmd_model
      - AWS_SHARED_CREDENTIALS_FILE=/secrets/aws_credentials
      - EPOCHS=2
```

Binding /app/data (i.e. the default DATA_DIR) to a persisted volume provides caching of steps within pipeline.  An operator can just delete the steps that require re-running vs. having to run full pipeline.  This is specifically useful when asset images haven't change but you want to re-run a model training with different parameters.  Just deleting the _trained_model_ folder would re-use previous downloaded and converted files but build and store a new model.
