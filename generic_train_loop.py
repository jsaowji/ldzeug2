#%%
from vstools import core, vs
from ldzeug2.vsnn import ModelType, Training, build_train_dataset
from ldzeug2.compact import compact
#%%
model = ModelType.COMPACT
dataset = build_train_dataset(model,90)
#%%
trn = Training(lambda : compact(1,1,addback=True),dataset)
#%%
trn.train()