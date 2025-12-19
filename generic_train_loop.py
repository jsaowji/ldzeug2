#%%
from vstools import core, vs
from ldzeug2.vsnn import ModelType, Training, build_train_dataset
from ldzeug2.compact import compact

model = ModelType.COMPACT
dataset = build_train_dataset(model,50, on_fields=False)
#%%
trn = Training(model.get_torch_model, dataset)
#%%
trn.train()
#%%
from matplotlib import pyplot as plt
import numpy as np
i,o = dataset
idx = 1
plt.imshow(np.array(i.get_frame(idx))[0])
plt.imshow(np.array(o.get_frame(idx))[0])
