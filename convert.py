from rknn.api import RKNN #Requires rknn-toolkit2 python api whl. Ubuntu x86 ONLY!

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[128, 128, 128], std_values=[128, 128, 128], target_platform='rk3588')
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model='best.onnx')
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
