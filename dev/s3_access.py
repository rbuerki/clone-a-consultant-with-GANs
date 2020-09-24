# This file was used for experimentation and debugging only.
import os
from configparser import ConfigParser
import boto3
from botocore.exceptions import ClientError, ParamValidationError


def config_s3(filename="dwh.cfg", section="AWS"):
    """Read and return necessary parameters for connecting to the s3 bucket."""
    # Create a parser to read config file
    parser = ConfigParser()
    parser.read(filename)

    # Get section, default to AWS
    s3_params = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            s3_params[param[0]] = param[1]
    else:
        raise Exception(f"Section {section} not found in the {filename} file.")

    return s3_params


def access_s3():
    """Acess to the Redshift cluster. Return bucket object."""
    try:
        # Read connection parameters
        s3_params = config_s3()
        # Connect to the PostgreSQL server
        key = s3_params.get('key')
        secret = s3_params.get('secret')

        s3 = boto3.resource(
            "s3",
            region_name="eu-west-1",
            aws_access_key_id=key,
            aws_secret_access_key=secret
        )
        s3_bucket = s3.Bucket("raph-dend-zh-data")
        # for obj in s3_bucket.objects.filter(Prefix="data/raw/verkehrszaehlungen/non_mot/"):
        #     print(obj)

    except ParamValidationError as e:
        print("Parameter validation error: %s" % e)
    except ClientError as e:
        print("Unexpected error: %s" % e)

    return s3_bucket


local_dir = "data/prep/"
bucket = "raph-dend-zh-data"
s3_dir = "data/prep/"


def upload_to_s3(local_dir, s3_bucket, s3_dir):
    """
    # Get local dir (from), S3 bucket and S3 dir (to) from CL
    # local_dir, bucket, s3_dir = sys.argv[1:4]
    """
    s3 = boto3.client('s3')
    # Enumerate local files recursively
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            # Construct the full local path
            local_path = os.path.join(root, filename)
            # Construct the full S3 path, set "/" as delimiter for key prefix
            relative_path = os.path.relpath(local_path, local_dir)
            if s3_dir is not None:
                s3_path = os.path.join(s3_dir, relative_path).replace("\\", "/")
            else:
                s3_path = relative_path.replace("\\", "/")
            # Check if file already exists, if not upload file
            try:
                s3.head_object(Bucket=bucket, Key=s3_path)
                print("File found on S3! Skipping %s..." % s3_path)
            except:
                print(f"Uploading {s3_path} ...")
                s3.upload_file(local_path, bucket, s3_path)
