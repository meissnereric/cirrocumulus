# Cloud Computing

### Credits:
Cloudformation template based off work in the AWS examples repo [https://github.com/aws-samples/aws-batch-processing-job-repo/blob/master/template/template.yaml](template).

## Architecture

The architecture for this works is as such:
* The primary compute for this work will be AWS Batch, which allows users to submit jobs to an auto-scaling compute cluster, so nodes are only spun up when there is work to do. AWS Batch leverages containers for the units of work, so we will be building a Dockerfile that has the required software dependencies.
* We will use S3 to write outputs from the compute jobs (i.e. trained model parameters), and also to read in data at the beginning of a job (you could also download it at the start of any job, but depending on the size of your data YMMV.)
* Jobs will be submitted from a simple looping script (```invoke_jobs.py```) that you can either run locally on your machine, or from an EC2 host that you've already spun up. This script can be upgraded to perform more complex operations like parallel bayesian optimization, but for this example we keep it simple.

### Files
* The Docker container definition is found in ```Dockerfile```
* The "core code", in this case the model training, is found in the ```train_model.py``` file. This file will be 
invoked from a shell script (```container_shell_script.sh```) first in the container. This is personal preference, as doing some installations or checks 
directly in shell script is simpler in a shell script than in python IMO. 
* The script to dispatch jobs to AWS Batch from your local host / EC2 instance is ```invoke_jobs.py```
* We use a cloudformation template (```template.yaml```) for creating all of the infrastructure we need in AWS.
 


## Cloudformation
Much of the AWS infrastructure for this is built in Cloudformation because its a nice way to create reproducable infrastucture, 'infrastructure-as-code'. This is a great thing, both because it saves you many button clicks on the GUIs, but also because in production you want to minimize the amount of places human error can come in, and the more parts of a process you can put through code review the better. Infra-as-code moves the infrastructure into your codebase and allows your team to code review it, and keep better track of when things are changing to isolate and reduce problems. 

The main components we will build here are:
* An AWS Batch compute cluster, job queue, and job definition (including networking infrastructure needed for the compute cluster)
* An S3 bucket for storing data our job may need, and for the job to write its output results to
* An ECR repository to store our custom Docker container 
* IAM roles and permissions setup for all of the above

## Local job script

The script ```invoke_jobs.py``` is used to invoke AWS Batch jobs. It sends jobs with different parameters (namely, the hyperparameters of different trainings of your model in this case) to the job queue. \

If you wanted to do implemenent the parallel BO loop described [here](http://proceedings.mlr.press/v84/kandasamy18a.html), 
then you would likely extend this script with that algorithm,
 continuing to utilize AWS Batch for the real computation and running this script from a separate EC2 machine for the central aggregation of results and dispatching of new jobs as the old ones complete. 

## Docker container + job definitions

## Container hook script + training script


