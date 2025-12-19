import numpy as np
import torch
import sys
from ldzeug2.compact import compact, compact_to_expr
from ldzeug2.experimentalyc import FullModelExperimental
from ldzeug2.colorcnn_trch import ColorCNNV2,ColorCNNV1

batch_size = 1

mdlpth = sys.argv[1]

#model = compact(num_in_ch=1, num_out_ch=1, num_feat=32, num_conv=14, upscale=1, kernel_size=3, act_type='prelu', bias=False)

model = compact(num_in_ch=1, num_out_ch=1, addback=True)
#model = FullModel()
#model = FullModel2(num_feat=64,num_conv=16)

x = torch.randn(batch_size, 1, 32, 32, requires_grad=True)
model.load_state_dict(torch.load(mdlpth))

model.eval()

torch_out = model(x)

torch.onnx.export(model,
                  (x),
                  mdlpth + ".onnx",
                  export_params=True,
                  do_constant_folding=True,
                  input_names  = ['input'],
                  output_names = ['output'], 
                  dynamic_axes={'input'  : {0 : 'batch_size', 2: "height", 3: "width"},
                                'output' : {0 : 'batch_size', 2: "height", 3: "width"},
                                })

print("exported")
