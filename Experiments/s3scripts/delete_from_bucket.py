import boto3
import argparse
from s3utils import delete_s3_obj

BUCKET_NAME = 'paralleladcexperiments5b70cd4e-74d3-4496-96fa-f4025220d48c'


def main(filename):
    s3_resource = boto3.resource('s3')
    delete_s3_obj(BUCKET_NAME, s3_resource, filename)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--filename' ,type=str)
    args = vars(arg_parser.parse_args())

    main(**args)