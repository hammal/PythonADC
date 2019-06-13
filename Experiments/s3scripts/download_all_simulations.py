import boto3
import argparse
from pathlib import Path
from s3utils import downloadFromS3, listObjectsInBucket


BUCKET_NAME = 'paralleladcexperiments5b70cd4e-74d3-4496-96fa-f4025220d48c'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pattern', type=str)
    parser.add_argument('-d', '--destination', type=str, default='./')
    args = vars(parser.parse_args())

    destination = Path(args['destination']).resolve()
    assert destination.exists(), "The directory '%s' does not exist" %destination

    s3_resource = boto3.resource('s3')
    objects = listObjectsInBucket(s3_resource, BUCKET_NAME, pattern=args['pattern'])
    for key in objects:
        downloadFromS3(s3_resource, BUCKET_NAME, key, destination)
