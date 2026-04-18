from rknn.api import RKNN

ONNX_MODEL = 'best_nano_111.onnx'
RKNN_MODEL = 'best_nano_111_rv1126b.rknn'
DATASET = 'dataset.txt'

# 根据你的训练输入尺寸改
IMG_SIZE = 640

rknn = RKNN(verbose=True)

print('--> Config model')
ret = rknn.config(
    target_platform='rv1126b',
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
)
if ret != 0:
    print('Config failed!')
    exit(ret)
print('done')

print('--> Load ONNX model')
ret = rknn.load_onnx(
    model=ONNX_MODEL
)
if ret != 0:
    print('Load ONNX failed!')
    exit(ret)
print('done')

print('--> Build model')
ret = rknn.build(
    do_quantization=True,
    dataset=DATASET
)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

print('--> Export RKNN model')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export RKNN failed!')
    exit(ret)
print('done')

rknn.release()
print('RKNN model generated:', RKNN_MODEL)
