import torch
import torch.nn as nn

def bn2instance(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.InstanceNorm2d(module.num_features,
                                                module.eps, module.momentum,
                                                module.affine,
                                                module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, bn2instance(child))

    del module
    return module_output

def bn2group(module):
    num_groups = 16 # hyper_parameter of GroupNorm
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.GroupNorm(num_groups,
                                           module.num_features,
                                           module.eps, 
                                           module.affine,
                                          )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, bn2group(child))

    del module
    return module_output
  

# import torchvision
# net = torchvision.models.resnet18()
# print(' \n ########################## Original ResNet18 ########################## \n',net)

# net = torchvision.models.resnet18()
# net = bn2instance(net)
# print(' \n ########################## ResNet18 with InstanceNorm ########################## \n',net)

# net = torchvision.models.resnet18()
# net = bn2group(net)
# print(' \n ########################## ResNet18 with GroupNorm ########################## \n',net)
