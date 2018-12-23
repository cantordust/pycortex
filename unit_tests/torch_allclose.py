import cortex.utest as utest
import torch

def test_torch_allclose(_add = 1e-6,
                        _rtol = 1e-7,
                        _atol = 1e-8):

    tensor1 = torch.ones(3,3)
    tensor2 = torch.ones(3,3)

    print("Tensor1:\n", tensor1)
    print("Tensor2:\n", tensor2)
    print("Relative tolerance:", _rtol)
    print("Absolute tolerance:", _atol)
    print("After adding", _add, "to element (0,0)")

    tensor2[0][0] += _add

    print("Tensor1 == Tensor2:", tensor1.allclose(tensor2, _rtol, _atol))

if __name__ == '__main__':
    utest.run(test_torch_allclose, 1e-6)
    utest.run(test_torch_allclose, 1e-7)
    utest.run(test_torch_allclose, 1e-8)
