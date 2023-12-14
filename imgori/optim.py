from mlconfig import register
from torch import optim

register(optim.Adam)
register(optim.RAdam)

register(optim.lr_scheduler.StepLR)
