from s3utils import listObjectsInBucket
import boto3


BUCKET_NAME = 'paralleladcexperiments5b70cd4e-74d3-4496-96fa-f4025220d48c'

def main():
    s3_resource = boto3.resource('s3')
    listObjectsInBucket(s3_resource, BUCKET_NAME)

if __name__ == "__main__":
    main()