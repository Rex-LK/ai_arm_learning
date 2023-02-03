import onnx
path='inference/onnx/rec_static.onnx'
model = onnx.load(path)
model.ir_version = 7
onnx.save(model, path)