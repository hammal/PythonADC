import boto3
import uuid
import io


BUCKET_NAME = 'paralleladcexperiments5b70cd4e-74d3-4496-96fa-f4025220d48c'



def create_s3_filename(file_name):
    return ''.join([str(uuid.uuid4().hex[:6]), file_name])


def writeLogToS3(s3_connection, bucket_name, file_name, logstr):
    string_buffer = io.StringIO()
    s3_filename = create_s3_filename(file_name)
    string_buffer.write(logstr)
    (s3_connection
      .Object(bucket_name, s3_filename)
      .put(Body=string_buffer.getvalue()))
    string_buffer.close()
    return s3_filename



with open('messages.log') as f:
    log = f.read()

print(type(log))
print(log)
file_name = "This_is_a_test_log"
s3_resource = boto3.resource('s3')
s3_file_name = writeLogToS3(s3_resource, BUCKET_NAME, file_name, log)