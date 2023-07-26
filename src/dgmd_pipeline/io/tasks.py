import os
import pathlib
import random
import boto3
import shutil
from statistics import mean

import luigi
from dgmd_pipeline.settings import AWS_DATA_BUCKET, PIPELINE_DATA_DIR, AWS_DATA_PATH


class DownloadTrainingFilesTask(luigi.Task):
    """Downloads files from CIRC GCS Bucket."""

    _data_dir = PIPELINE_DATA_DIR + "/raw"
    _aws_bucket_name = AWS_DATA_BUCKET
    _aws_data_path = AWS_DATA_PATH

    def output(self):
        return luigi.LocalTarget(self._data_dir)

    def download_dir(
        self,
        client,
        resource,
        dist,
        local="/tmp",
        bucket="your_bucket",
        filter_path=None,
    ):
        paginator = client.get_paginator("list_objects")
        for result in paginator.paginate(Bucket=bucket, Delimiter="/", Prefix=dist):
            if result.get("CommonPrefixes") is not None:
                for subdir in result.get("CommonPrefixes"):
                    self.download_dir(
                        client,
                        resource,
                        subdir.get("Prefix"),
                        local,
                        bucket,
                        filter_path,
                    )
            for file in result.get("Contents", []):
                dest_pathname = os.path.join(local, file.get("Key"))

                # Remove extra file paths if needed:
                if filter_path:
                    dest_pathname = dest_pathname.replace(filter_path, "")

                # Download files:
                if not os.path.exists(os.path.dirname(dest_pathname)):
                    os.makedirs(os.path.dirname(dest_pathname))
                if not file.get("Key").endswith("/"):
                    resource.meta.client.download_file(
                        bucket, file.get("Key"), dest_pathname
                    )

    def run(self):
        client = boto3.client("s3")
        resource = boto3.resource("s3")
        self.download_dir(
            client,
            resource,
            self._aws_data_path,
            self._data_dir,
            bucket=self._aws_bucket_name,
            filter_path=self._aws_data_path,
        )


class LimitTrainingFiles(luigi.Task):
    """Limits the number of files that are used to some max.

    Used to prevent over-balanced sampling if uneven category sets.
    """

    _data_dir = PIPELINE_DATA_DIR + "/limited"
    _upper_threshold = 1.15

    def requires(self):
        return DownloadTrainingFilesTask()

    def output(self):
        return luigi.LocalTarget(self._data_dir)

    def run(self):
        input_folder = self.input().path

        min_file_cnt = (
            mean([len(files) for r, d, files in os.walk(input_folder)])
            * self._upper_threshold
        )

        # Iterate through all files and seperate:
        for root, dirs, files in os.walk(input_folder):
            sub_dir = pathlib.PurePath(root).name
            for filename in random.sample(files, min(len(files), min_file_cnt)):
                destination_dir = os.path.join(self._data_dir, sub_dir)
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir, exist_ok=True)

                # Copy file to assigned folder:
                shutil.copy(
                    os.path.join(root, filename),
                    os.path.join(destination_dir, filename),
                )
