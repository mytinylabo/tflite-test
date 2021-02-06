FROM python:3.7.9

# docker build -t tflite-test
# docker run -it --rm -v $PWD:/workspace tflite-test

RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
RUN pip install numpy pillow click

WORKDIR /workspace
ENTRYPOINT /bin/bash
