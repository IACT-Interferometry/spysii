from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='spysii',
    version='0.1',
    packages=['spysii', 'spysii.data'],
    # packages=find_packages(),
    install_requires=[
        'click',
        'numpy>=1.23.0,<1.25.0',
        'scipy',
        'matplotlib',
        'astropy',
        'lmfit',
        'emcee',
        'click',
        'coloredlogs',
        'oifits',
        # 'yaml',
    ],
    include_package_data=True,
    author="Tarek Hassan",
    author_email="tarek.hassan@ciemat.es",
    description="A software in python for stellar intensity interferometry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TarekHC/spysii",
    # packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
 )
