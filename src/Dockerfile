FROM pytorch/pytorch


COPY container_shell_script.sh train_model.py ./


RUN pip install --upgrade pip && \
    pip install boto3 && \
    pip install boto 

RUN pwd
RUN ls

#CMD ["echo", "hello world"]
CMD ["bash", "container_shell_script.sh"]
