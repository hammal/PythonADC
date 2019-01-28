import boto3


BUCKET_NAME = 'paralleladcexperiments5b70cd4e-74d3-4496-96fa-f4025220d48c'


def listObjectsInBucket(bucket_name, s3_connection):
    bucket = s3_connection.Bucket(name=bucket_name)
    for obj in bucket.objects.all():
        print(obj.key)


def main():
    s3_resource = boto3.resource('s3')
    listObjectsInBucket(BUCKET_NAME, s3_resource)

if __name__ == "__main__":
    main()