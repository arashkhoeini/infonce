from setuptools import setup, find_packages

setup(
    name="infonce",
    version="0.2",
    packages=find_packages(),
    install_requires=[  
        "torch>=2.0.0",
    ],
    author="Arash Khoeini",
    author_email="arash.khoeini@gmail.com",
    description="Pytorch-based InfoNCE loss for self-supervised learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/arashkhoeini/infonce",  # GitHub or project link
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)