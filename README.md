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

### Setting up your IAM user credentials locally
Before we can call AWS Batch from the command line or locally using python (which is what the invoke_jobs.py file does), we need to configure it with some credentials so that it can access our account and resources.

#### Creating an IAM user
IAM users are a way to manage the permissions of different users / applications in AWS. When you call the AWS API, you'll do some using a User's credentials, and each user can be a part of different policy groups, or have policies that allow that user to do things like spin up EC2 instances, or start AWS Batch jobs.

To create a new user for this example:
1. Navigate to the IAM service page from the AWS Console page.
2. Click on Users on the lefthand side
3. Click 'Add user'
4. Give it a meaningful username, i.e. 'cloud-compute', and check the 'programmatic access' tickbox. Click next.
5. For permissions, click on the "Attach existing policies directly" tab, then search for 'AWSBatchFullAccess' and tick it's box. Also add the 'AmazonS3FullAccess' policy.
6. Click through tags, and then click create user.
7. **Important** Make sure you download the CSV with your access + secret access keys here and keep it in a safe place. Once you create the user, you can't get the secret access key again if you lose it (You can rotate these keys later on, but that invalidates the old keys that might have been in use.)

#### Adding your user credentials to the AWS CLI
Take your access key, and secret access key, and run the
```bash
aws configure
```
command from a terminal. This should ask for these keys and a default region to use (us-east-1 is the default, but eu-west-1 is the nearest. Make sure whichever you use is the same region you create your Cloudformation stack / other resources in. It doesn't much matter usually, but different regions sometimes have different services avaiable.)



### How to use the script
The script ```invoke_jobs.py``` is used to invoke AWS Batch jobs. It sends jobs with different parameters (namely, the hyperparameters of different trainings of your model in this case) to the job queue. \

If you wanted to do implemenent the parallel BO loop described [here](http://proceedings.mlr.press/v84/kandasamy18a.html), 
then you would likely extend this script with that algorithm,
 continuing to utilize AWS Batch for the real computation and running this script from a separate EC2 machine for the central aggregation of results and dispatching of new jobs as the old ones complete. 

## Docker container + job definitions
### Building your container
If you're in the same directory as your Dockerfile, you can run the following to build and tag your image:
```bash
docker build --tag my-tag .
```
You can then run this image as a container via:
```bash
docker run --name my-container my-tag:1.0
```

You can pass a command to override the default container command by simply passing the command as the last argument.
Adding the '-it' parameter will drop you into the running container's shell, and the 'rm' command removes the container once you close the container.:
```bash
    docker run -it --rm --name my-container my-tag:1.0 top
```

### Uploading your built container to ECR
To upload the container you've built to ECR (Elastic Container Repository) run:
```bash
docker images
docker tag <image_id> <aws_account_id>.dkr.ecr.<region>.amazonaws.com/<my-container-repo>
docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/<my-container-repo>
```
where image_id is the local image ID, region is the AWS region code (i.e. us-east-1, eu-west-1, etc.), and 'my_container_repo' is the ECR repository (i.e.  batch-processing-job-repository from template.yaml)

## Container hook script + training script
So there are two scripts that we add to the Docker image we upload and run. 

These are the ```container_shell_script.py``` and ```train_model.py```.

You could get away with only have one file, or uploading many more, but this is a standard model I use.

The shell script we will use to install dependencies (apt-get, pip, git clone and install your personal code repository),
 and the train_model file is where we will run the actual training code for what we want to do.

We pass parameters to these via the ```parameters``` variable in the ```submit_jobs.py``` file, so if you wanted to pass more parameters such as the location of a dataset to download, you might add that to the parameters variable there. 

