import setuptools

setuptools.setup(name='cortex',
                 version='0.1.1',
                 description='Library for evolving deep learning models using PyTorch',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 install_requires=['numpy','colorama','tensorboardX', 'mpi4py'])
