import configparser
from pathlib import Path
from typing import Iterator, Tuple, Union

import boto3
from botocore.exceptions import ClientError


def parse_config_img_data(
    path_to_cfg: str,
) -> Tuple[str, str, str, str, str, str]:
    """Retrieve necessary credentials for AWS access. (Btw. see
    `dev` folder for a more elegant way to parse config files.)
    """
    config = configparser.ConfigParser()
    try:
        config.read(path_to_cfg)
    except FileNotFoundError as e:
        print(f"Please check the path to the config file: {e}")
        raise

    aws_access_key_id = config.get("AWS", "ACCESS_KEY")
    aws_secret_access_key = config.get("AWS", "SECRET_KEY")
    path_to_upload_data = config.get("S3_UPLOAD", "LOCAL_RELPATH")
    s3_region = config.get("S3_UPLOAD", "REGION")
    s3_bucket = config.get("S3_UPLOAD", "BUCKET")
    s3_prefix = config.get("S3_UPLOAD", "PREFIX_S3_NAME")
    return (
        aws_access_key_id,
        aws_secret_access_key,
        path_to_upload_data,
        s3_region,
        s3_bucket,
        s3_prefix,
    )


def instantiate_s3_client(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_region: str = "eu-west-1",
) -> boto3.client:
    """Instantiate the s3 client using the provided credentials
    and passing a region of choice.
    """
    try:
        s3_client = boto3.client(
            "s3",
            region_name=s3_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        return s3_client
    except ClientError as e:
        print(f"Could not instantiate s3 client: {e}")
        raise


def check_for_s3_bucket(
    s3_client: boto3.client, s3_bucket: str, s3_region: str = "eu-west-1"
):
    """Check if the given bucket already exists in the s3 account.
    If not create the bucket.
    """
    my_buckets = s3_client.list_buckets()
    if s3_bucket not in [bucket["Name"] for bucket in my_buckets["Buckets"]]:
        try:
            s3_client.create_bucket(
                Bucket=s3_bucket,
                CreateBucketConfiguration={"LocationConstraint": s3_region},
            )
        except ClientError as e:
            print(f"Bucket creation failed: {e}")
            raise


def create_file_generator(
    path_to_upload_data: Union[str, Path]
) -> Iterator[Path]:
    """Walk the directory tree at given path and return a generator
    object containing all the file path objects.
    """
    return Path(path_to_upload_data).rglob("*.*")


def upload_file_to_s3(
    local_path: Path, s3_client: boto3.client, s3_bucket: str, s3_prefix: str,
) -> bool:
    """Upload a file to the given s3 bucket using the given client, if
    it don't alread exists there. There is an option to add a 'prefix'
    (extra parent dirs) to the s3 filename, for example if you
    only want to upload a part of the original local data directory.
    By default this prefix string is empty (see config file).
    """
    s3_path = str(Path(s3_prefix) / local_path).replace("\\", "/")
    # Check if file already exists, if yes skip, if no upload
    try:
        s3_client.head_object(Bucket=s3_bucket, Key=s3_path)
        print(f"File found in s3 bucket! Skipping {s3_path}")
    except ClientError:
        try:
            print(f"Uploading {s3_path} ...")
            s3_client.upload_file(str(local_path), s3_bucket, s3_path)
            return True
        except ClientError as e:
            print(f"Upload failed: {e}")
            raise


def main(path_to_cfg: str):
    """Run upload pipeline."""
    count_upload = 0
    (
        aws_access_key_id,
        aws_secret_access_key,
        path_to_upload_data,
        s3_region,
        s3_bucket,
        s3_prefix,
    ) = parse_config_img_data(path_to_cfg)

    s3_client = instantiate_s3_client(aws_access_key_id, aws_secret_access_key)
    check_for_s3_bucket(s3_client, s3_bucket, s3_region)
    file_generator = create_file_generator(path_to_upload_data)
    for file in file_generator:
        response = upload_file_to_s3(file, s3_client, s3_bucket, s3_prefix)
        count_upload += response
    print(f"\nSuccessfully uploaded {count_upload} files.")


if __name__ == "__main__":
    main("config.cfg")
