import boto3
import argparse

parser = argparse.ArgumentParser(description='Process input parameters..')
parser.add_argument('net_width', metavar='N', type=int, nargs='+',
                    help='the network width')

args = parser.parse_args()

input_parameter = args['net_width']
sin_value = 1

msg = "{} : {}".format(input_parameter, sin_value)

client = boto3.client('sqs')

queue='cloud-compute-sqs'

response = client.send_message(
    QueueUrl=queue,
    MessageBody=str(sin_value))
