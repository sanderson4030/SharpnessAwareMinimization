# Sharpness-Aware Minimization

Unofficial implementation of [Sharpness-Aware Minimization (SAM)](https://arxiv.org/abs/2010.01412) (Proceedings of ICLR 2021) for [fast.ai (V2)](https://docs.fast.ai/).

This package provides the SAM (Sharpness-Aware Minimization) callback for use with the fastai learner.


## Usage

To use SAM you need to import `sam` and pass the corresponding callback to the 'cbs' argument when calling a .fit() function :
```python
from sam import SAM
learn.fit_one_cycle(1, 3e-4, wd=.1, cbs=SAM(rho=.05))
```
### SAM

SAMC has only one parater: `rho`

`rho` is a hyperparameter controling the distance of the virtual step size used in SAM. The default setting for `rho` is 0.05, but this will not always be the ideal setting. The authors recomend performing a grid search over the following range to find the best value for your model and data:
{0.01, 0.02, 0.05, 0.1, 0.2, 0.5}

Each step while using SAM requires two passes over each batch, in most cases causing 2x slowdown during training. 
SAM also uses more memory during batches due to an additional copy of the model and gradients being stored during the backard pass. 


*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*
