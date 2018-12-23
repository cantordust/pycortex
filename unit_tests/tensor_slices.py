import cortex.utest as utest

import torch

def test_tensor_slices():

    tensor = torch.randn(5,3,3)
    new_tensor = torch.Tensor()

    print(tensor)
    print(new_tensor)

    slices = [slice(0,2), slice(3,None)]
    for slice_index in range(len(slices)):
        if slice_index == 0:
            new_tensor = tensor[slices[slice_index]]
        else:
            new_tensor = torch.cat((new_tensor, tensor[slices[slice_index]]))

    print(tensor)
    print(new_tensor)

if __name__ == '__main__':
    utest.run(test_tensor_slices)