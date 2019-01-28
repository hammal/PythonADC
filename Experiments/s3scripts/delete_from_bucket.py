import boto3
import argparse

BUCKET_NAME = 'paralleladcexperiments5b70cd4e-74d3-4496-96fa-f4025220d48c'


def delete_s3_obj(bucket_name, s3_connection, filename):
    obj = s3_connection.Object(bucket_name, filename)
    obj.delete()


def main(filename):
    s3_resource = boto3.resource('s3')
    delete_s3_obj(BUCKET_NAME, s3_resource, filename)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-f', '--filename' ,type=str)
    args = vars(arg_parser.parse_args())

    main(**args)