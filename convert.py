import numpy as np
from rknn.api import RKNN

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[128, 128, 128], std_values=[128, 128, 128], target_platform='rk3588')
    print('done')

    # Load model (from https://www.tensorflow.org/lite/guide/hosted_models?hl=zh-cn)
    print('--> Loading model')
    ret = rknn.load_tflite(model='model.tflite')
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn('model.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
