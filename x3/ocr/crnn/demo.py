import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import onnx
from onnxsim import simplify
import models.crnn as crnn


model_path = './data/crnn.pth'
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())

export_onnx = 1
if export_onnx:
    torch.onnx.export(
        model, (image,),
        "crnn.onnx",
        input_names=["image"],
        output_names=["output"],
        opset_version=11
    )
    input_path="crnn.onnx"
    output_path="crnn_simp.onnx"
    onnx_model = onnx.load(input_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('finished exporting onnx')

preds = model(image)
_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)
preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
