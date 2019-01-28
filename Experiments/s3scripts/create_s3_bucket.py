import boto3
import uuid


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


def listObjectsInBucket(bucket_name, s3_connection):
    bucket = s3_connection.Bucket(name=bucket_name)
    for obj in bucket.objects.all():
        print(obj.key)


def upload_to_s3(bucket_name, s3_connection, filename):
    bucket = s3_connection.Bucket(bucket_name)
    bucket.upload_file(Filename=filename,Key=filename)



def main():
    s3_resource = boto3.resource('s3')

    # with open('bucket_config.txt') as f:
    #     bucket_name = f.read().strip('\n')

    # upload_to_s3(bucket_name, s3_resource, 'bucket_config.txt')
    # listObjectsInBucket(bucket_name, s3_resource)
    bucket_name, bucket_response = create_s3_bucket(
        prefix='paralleladcexperiments', s3_connection=s3_resource)

    with open('bucket_name.txt','w') as f:
        f.write(bucket_name)
        # f.write(bucket_response)

if __name__ == "__main__":
    main()