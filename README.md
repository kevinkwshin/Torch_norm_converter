# Torch_norm_converter
Change Batchnorm to InstanceNorm or GroupNorm of the Pytorch Model

```python
import converter
import torchvision

net = torchvision.models.resnet18()
print(' \n ########################## Original ResNet18 ########################## \n',net)

net = torchvision.models.resnet18()
net = converter.bn2instance(net)
print(' \n ########################## ResNet18 with InstanceNorm ########################## \n',net)

net = torchvision.models.resnet18()
net = converter.bn2group(net)
print(' \n ########################## ResNet18 with GroupNorm ########################## \n',net)
```
