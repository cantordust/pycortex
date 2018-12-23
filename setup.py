import setuptools

setuptools.setup(name='cortex',
                 version='0.1.1',
                 description='Library for evolving deep learning models in PyTorch',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 install_requires=['torch','torchvision','numpy','colorama','tensorboardX'])
