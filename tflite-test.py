import time
import click
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite


def get_label(result):
    with open('model/labels_mobilenet_quant_v1_224.txt', 'r') as f:
        labels = f.read().splitlines()
    return labels[np.argmax(result)]


@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.option('-n', '--n-runs', 'n_runs', type=int, default=10)
def process(input_file, n_runs):
    # モデルをロード
    tflite_model = 'model/mobilenet_v1_1.0_224_quant.tflite'

    interpreter = tflite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    print(repr(input_details[0]['shape']))

    # 画像をロード
    input_image = np.array(Image.open(input_file).resize((224, 224)))
    input_image = np.expand_dims(input_image, axis=0)

    # 推論器に画像をセット
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # 推論実行
    start_time = time.time()
    for i in range(0, n_runs):
        interpreter.invoke()
    end_time = time.time()

    # 平均時間を算出
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print('Predicted value:', np.argmax(output_data))
    print('Predicted label:', get_label(output_data))

    print('----------------------')
    print('Average time: %f ms' % ((end_time - start_time) * 1000 / n_runs))


def main():
    process()


if __name__ == "__main__":
    main()
