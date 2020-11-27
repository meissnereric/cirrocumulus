# Make sure you have configured your AWS credentials in the command line before using Boto3.
import boto3

sqs_client = boto3.client('sqs')
client = boto3.client('batch')

widths=[100,200,400]

sqsQueue= 'cloud-compute-sqs'
jobQueue = 'BatchProcessingJobQueue'
jobDefinition = 'BatchJobDefinition:1'

for width in widths:
    jobName = 'batch-job-{}'.format(width) 
    parameters = {'width' : str(width)}
    client.submit_job(jobName = jobName,
                      parameters = parameters,
                      jobQueue = jobQueue,
                      jobDefinition = jobDefinition)

        
# Use something like the following lines to listen for your jobs
# to respond with their results if doing Parallel BO
#
#dataset = {}
#
## Here 'run_values' would be the next set of datapoints you want to test with the jobs.
## If we wanted to do this above, it would be 'widths' since thats what we're running the jobs with.
#while run_values not in dataset:
#    while response is None:
#        response = client.receive_message(
#            QueueUrl=sqsQueue)
#    for message in response['Messages']:
#        input_value, output_value = message['Body']
#        dataset[input_value] = output_value
   
