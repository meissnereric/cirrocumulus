FROM pytorch/pytorch


COPY container_shell_script.sh train_model.py ./

RUN pip install --upgrade pip && \
    pip install boto3 && \
    pip install boto 

RUN git clone https://github.com/wehrley/Kaggle-Digit-Recognizer.git
RUN git clone https://github.com/meissnereric/cirrocumulus.git

RUN pwd
RUN ls

#CMD ["echo", "hello world"]
CMD ["bash", "container_shell_script.sh"]
