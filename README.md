## Install dependencies
```sh
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
pip install numpy pillow click
```

... or use Docker:
```sh
docker build -t tflite-test
docker run -it --rm -v $PWD:/workspace tflite-test
```

## Usage
```sh
python tflite-test.py images/zebra.jpg
```

## See also
- https://www.tensorflow.org/lite/guide/get_started
- https://www.tensorflow.org/lite/models/image_classification/overview
