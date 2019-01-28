import boto3
import uuid
import argparse


BUCKET_NAME = 'paralleladcexperiments5b70cd4e-74d3-4496-96fa-f4025220d48c'


def downloadFromS3(s3_connection, bucket_name, file_name, destination):
    (s3_connection
        .Object(bucket_name, file_name)
        .download_file(f'{destination}/{file_name}'))


def main(file_name, destination):
    s3_resource = boto3.resource('s3')
    downloadFromS3(s3_resource, BUCKET_NAME, file_name, destination)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--file_name', type=str)
    arg_parser.add_argument('-d', '--destination', type=str)
    args = vars(arg_parser.parse_args())
    
    main(**args)