import boto3
import uuid
import argparse
import re


BUCKET_NAME = 'paralleladcexperiments5b70cd4e-74d3-4496-96fa-f4025220d48c'


def create_s3_filename(file_name):
    return ''.join([str(uuid.uuid4().hex[:6]), file_name])


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
    print("Created bucket: %s\nRegion: %s" % (bucket_name, current_region))
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


def writeCSVDataFrameToS3(s3_connection, bucket_name, file_name, df):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    (s3_connection
      .Object(bucket_name, file_name)
      .put(Body=csv_buffer.getvalue()))
    string_buffer.close()


def writeStringToS3(s3_connection, bucket_name, file_name, string):
    string_buffer = io.StringIO()
    string_buffer.write(string)
    (s3_connection
      .Object(bucket_name, file_name)
      .put(Body=string_buffer.getvalue()))
    string_buffer.close()


def uploadTos3(s3_connection, bucket_name, file_name, obj):
    pickle_buffer = io.BytesIO()
    pkl.dump(obj, pickle_buffer)
    (s3_connection
      .Object(bucket_name, file_name)
      .put(Body=pickle_buffer.getvalue()))
    pickle_buffer.close()