import boto3

client = boto3.client('batch')

widths=[100,200,400]

jobQueue = 'BatchProcessingJobQueue'
jobDefinition = 'BatchJobDefinition:1'

for width in widths:
    jobName = 'batch-job-{}'.format(width) 
    parameters = {'width' : str(width)}
    client.submit_job(jobName = jobName,
                      parameters = parameters,
                      jobQueue = jobQueue,
                      jobDefinition = jobDefinition)
    
