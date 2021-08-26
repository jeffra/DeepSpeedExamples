FROM nvcr.io/nvidia/pytorch:21.02-py3

RUN mkdir /sc21-code

ADD megatron-small-indexed /sc21-code/data

RUN apt-get update && \
    apt-get install -y libaio-dev ninja-build

RUN git clone https://github.com/jeffra/DeepSpeed.git /sc21-code/DeepSpeed && \
    cd /sc21-code/DeepSpeed && \
    git checkout sc21i && \
    pip install . && \
    ds_report

RUN git clone https://github.com/jeffra/DeepSpeedExamples.git /sc21-code/DeepSpeedExamples && \
    cd /sc21-code/DeepSpeedExamples && \
    git checkout sc21i

# open permissions on code and data
RUN chmod -R 777 /sc21-code
