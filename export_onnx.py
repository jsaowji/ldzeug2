import numpy as np
import torch
import sys
from ldzeug2.compact import compact
from ldzeug2.experimentalyc import FullModelExperimental
from ldzeug2.colorcnn_trch import FullModel2,FullModel

batch_size = 1

mdlpth = sys.argv[1]

model = compact(num_in_ch=1, num_out_ch=1, num_feat=32, num_conv=14, upscale=1, kernel_size=3, act_type='prelu', bias=False)

#model = compact(num_in_ch=1, num_out_ch=1, num_feat=8, num_conv=10, upscale=1, kernel_size=3, act_type='prelu',bias=False)
#model = FullModel()
#model = FullModel2(num_feat=64,num_conv=16)
#model = FullModelExperimental()

x = torch.randn(batch_size, 1, 32, 32, requires_grad=True)
#x = torch.randn(batch_size, 3, 32, 32, requires_grad=True)

#model = experimental(num_in_ch=2, num_out_ch=1, num_feat=64, num_conv=16, upscale=1, kernel_size=3, act_type='prelu', bias=False)
#x = torch.randn(batch_size, 2, 32, 32, requires_grad=True)


#model.load_state_dict(torch.load(mdlpth)["params"])
model.load_state_dict(torch.load(mdlpth))

model.eval()



#x = torch.randn(batch_size, 1, 32, 32, requires_grad=True)



torch_out = model(x)

torch.onnx.export(model,
                  (x),
                  sys.argv[2],
                  export_params=True,
                  do_constant_folding=True,
                  input_names  = ['input'],
                  output_names = ['output'], 
                  dynamic_axes={'input'  : {0 : 'batch_size', 2:"height", 3:"width"},
                                'output' : {0 : 'batch_size', 2:"height", 3:"width"},
                                })

print("exported")
