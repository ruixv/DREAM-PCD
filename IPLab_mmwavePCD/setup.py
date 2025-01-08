from setuptools import setup, find_packages

# Read requirements.txt content
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="IPLab_mmwavePCD",
    version="0.1",
    packages=find_packages(),
    install_requires=required + [
        # PyTorch is not specified here as it requires specific versions (CPU/CUDA)
        # Please install PyTorch separately according to your environment
    ],
    author="Ruixu Geng",
    author_email="gengruixu@mail.ustc.edu.cn",
    description="A Python package for mmWave radar point cloud processing",
    url="https://github.com/ruixv/RadarEyes",
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)