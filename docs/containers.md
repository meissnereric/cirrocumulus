# Cloud Computing

## Virtualization
One of the most basic building blocks that you'll use in cloud computing is a *virtualized instance*. Essentially, this is a virtual machine that the cloud provider runs on top of some *bare metal* server, and when you spin up an EC2 instance, you are running inside of that virtual machine. What you see is a full computing environment, you choose the base image that the virtual machine is spun up with. The image is primarily the operating system, so what linux distribution/version, but can also include pre-installed things like drivers or system libraries that are useful. 

For all intents and purposes, you don't really need to care that your instance is virtualized, it will behave almost identically to if you were running directly on a server running the same OS. You can SSH to the host, install packages, etc. just like normal to an EC2 host (or equivalent in other cloud providers.)

## Docker
### Why use it
So Docker, which we use interchangably with  "containerization" and "containers" as it is really the industry standard for containers, at its simplest is just a way to package up the:
* code
* code dependencies
* system requirements

for a piece of software into a *static* artifact that can be deployed onto any host (Windows, Linux, Mac, etc.) that runs Docker. I emphasize that this is a static artifact because each time you build a container, it gives that build artifact a particular ID, such that future applications that build on top of that container use precisely that build environment.

### Building a container
You can build containers out of (on top of) other containers people have built. For instance, you might want to base your container off the ```pytorch``` container, which is in turn built on top of the ```ubuntu```. Each of these containers in the chain is a dependency, so like other software dependencies it is very helpful that they are versioned. That's really all a Docker container is doing, it saves a snapshot of some environment (more precisely, the steps it took to get to some state), and gives that a name.

Let's look at a basic Dockerfile.
```dockerfile
FROM pytorch/pytorch

COPY container_shell_script.sh train_model.py /


RUN pip install --upgrade pip && \
    pip install boto3 && \
    pip install boto

RUN pwd
RUN ls

CMD ["bash", "container_shell_script.sh"]
```

The first line
```dockerfile
FROM pytorch/pytorch
```
determines what base container we will build from. The Docker process goes to [Dockerhub](https://hub.docker.com/r/pytorch/pytorch) and finds the specified container and downloads it (and all of that containers chain of dependenies.)

Next we copy some files from our local directory that we will need inside the container using the ```COPY``` command.

We can run commands inside the container, such as the pip install steps here, with ```RUN```. 

The ```CMD``` keyword sets a default command that runs when the container is run, and can be overrided from the command line. We indicate here that, if not specified otherwise, that when the container is spun up it should run the ```container_shell_script.sh``` file using ```bash```.

See the [Docker documentation](https://docs.docker.com/) for further information on commands and how to use them.
