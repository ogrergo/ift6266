from dataset import load_data
from deconvolution import build_deconv_function

trainset_x, trainset_y = load_data(0, 128)
f = build_deconv_function()

trainset_x = trainset_x.transpose((0,3,1,2))
r = f(trainset_x)

print(r)