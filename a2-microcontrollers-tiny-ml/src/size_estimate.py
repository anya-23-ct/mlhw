import torch
import numpy as np
import torch.nn as nn


def flop(model, input_shape, device):
    total = {}

    def count_flops(name):
        def hook(module, input, output):
            "Hook that calculates number of floating point operations"
            flops = {}
            batch_size = input[0].shape[0]
            if isinstance(module, nn.Linear):
                # TODO: fill-in (start)
                # raise NotImplementedError
                input_features = input[0].shape[1]
                output_features = output.shape[1]
                weight_flops = 2 * input_features * output_features      
                bias_flops = output_features if module.bias is not None else 0
                total_flops = batch_size * (weight_flops + bias_flops)
                flops[module] = total_flops
                # TODO: fill-in (end)

            if isinstance(module, nn.Conv2d):
                # TODO: fill-in (start)
                # raise NotImplementedError
                kernel_size = module.kernel_size[0] * module.kernel_size[1] 
                input_channels = input[0].shape[1]
                output_channels = output.shape[1]
                output_height = output.shape[2]
                output_width = output.shape[3]
                kernel_ops = 2 * kernel_size * input_channels
                total_flops = batch_size * kernel_ops * output_channels * output_height * output_width
                if module.bias is not None:
                    total_flops += batch_size * (output_channels * output_height * output_width)
                flops[module] = total_flops
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm1d):
                # TODO: fill-in (end)
                # raise NotImplementedError
                num_elements = input[0].numel()
                num_channels = input[0].shape[1]
                total_flops = 2 * num_channels * batch_size * num_elements  # Two operations per element
                flops[module] = total_flops
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm2d):
                # TODO: fill-in (end)
                # raise NotImplementedError
                num_elements = input[0].numel()
                num_channels = input[0].shape[1]
                num_spacial_elements = input[0].shape[2] * input[0].shape[3]
                total_flops = 2 * num_channels * batch_size * num_spacial_elements * num_elements
                flops[module] = total_flops
                # TODO: fill-in (end)
            total[name] = flops
        return hook

    handle_list = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(count_flops(name))
        handle_list.append(handle)
    input = torch.ones(input_shape).to(device)
    model(input)

    # Remove forward hooks
    for handle in handle_list:
        handle.remove()
    return total


def count_trainable_parameters(model):
    """
    Return the total number of trainable parameters for [model]
    :param model:
    :return:
    """
    # TODO: fill-in (start)
    # raise NotImplementedError
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
    # TODO: fill-in (end)


def compute_forward_memory(model, input_shape, device):
    """

    :param model:
    :param input_shape:
    :param device:
    :return:
    """
    
    # TODO: fill-in (start)
    # raise NotImplementedError
    input = torch.ones(input_shape, device=device)
    model = model.to(device)
    output = model(input)
    output_mem = output.numel() * output.element_size()
    input_mem = input.numel() * input.element_size()
    return input_mem + output_mem
    # TODO: fill-in (end)

