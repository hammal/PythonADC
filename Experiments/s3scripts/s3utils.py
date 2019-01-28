import boto3
import uuid
import argparse
import re


BUCKET_NAME = 'paralleladcexperiments5b70cd4e-74d3-4496-96fa-f4025220d48c'


def downloadFromS3(s3_connection, bucket_name, file_name, destination):
    (s3_connection
        .Object(bucket_name, file_name)
        .download_file(f'{destination}/{file_name}'))


def delete_s3_obj(bucket_name, s3_connection, filename):
    obj = s3_connection.Object(bucket_name, filename)
    obj.delete()


def create_bucket_name(prefix):
    return ''.join([prefix, str(uuid.uuid4())])


def create_s3_bucket(prefix, s3_connection):
    session = boto3.session.Session()
    current_region = session.region_name
    bucket_name = create_bucket_name(prefix)
    bucket_response = s3_connection.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={
        'LocationConstraint': current_region})
    print(bucket_name, current_region)
    return bucket_name, bucket_response


def listObjectsInBucket(s3_connection, bucket_name, pattern=r'.*'):
    regex = re.compile(pattern)
    bucket = s3_connection.Bucket(name=bucket_name)
    objects = []
    for obj in bucket.objects.all():
        if regex.match(obj.key):
            objects.append(obj.key)
        else:
            continue
    return objects

