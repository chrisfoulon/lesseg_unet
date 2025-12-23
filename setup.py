from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='lesseg_unet',        # This is the name of your PyPI-package.
    version='2.0.3',     # Fixed auto-config batch_size vs fold size constraint
    python_requires='>=3.11',  # Required by scipy 1.16.3
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(exclude=['__pycache__']),
    # Updated requirements for PyTorch 2.7 compatibility
    install_requires=[
        # PyTorch ecosystem
        'torch==2.7.0',
        'torchvision==0.22.0',
        'monai==1.5.1',
        'torchio==0.21.0',
        # Neuroimaging libraries
        'nibabel==5.3.3',
        'nilearn==0.12.1',
        # Scientific computing
        'scipy==1.16.3',
        'pandas==2.3.3',
        # Visualization
        'matplotlib==3.10.8',
        'seaborn==0.13.2',
        'tensorboard==2.20.0',
        # Utilities
        'tqdm==4.67.1',
        'einops==0.8.1',  # Required for UNETR/SwinUNETR
        'python-dateutil==2.9.0.post0',
        'dask==2025.12.0',
        # Custom package
        'bcblib>=0.3.4.3',
    ],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst", "*.md"],
        # Include all the data
        "data": ["*.*"],
    },
    # https://reinout.vanrees.org/weblog/2010/01/06/zest-releaser-entry-points.html
    # entry_points could be used to automagically download dcm2niix depending on the OS of the user
    entry_points={
        'console_scripts': ['lesseg_unet = lesseg_unet.main:main']
        # 'console_scripts': ['dicom_conversion = data_identification.scripts.dicom_conversion:convert']
    },
    # metadata to display on PyPI
    author="Chris Foulon",
    author_email="c.foulon@ucl.ac.uk",
    description="This project is about identifying the type of image and correcting headers during "
                "DICOM to nifti conversion",
    long_description=read('README.md'),
    keywords='Unet lesion segmentation mri dwi',
    url="https://scm4.cs.ucl.ac.uk/Foukalas/data-identification-and-curation",  # project home page, if any
    project_urls={
        "Wiki": "https://scm4.cs.ucl.ac.uk/Foukalas/data-identification-and-curation/-/wikis/home",
    },
    classifiers=[
        "License :: OSI Approved :: Python Software Foundation License"
    ]
    )
