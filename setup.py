from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='lesseg_unet',        # This is the name of your PyPI-package.
    version='0.10.7',     # Update the version number for new releases
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(exclude=['__pycache__']),
    # einops is only for the UNETR
    install_requires=['nibabel>=3.2.2', 'numpy>=1.18.5', 'nilearn>=0.9.0', 'monai>=1.0', 'bcblib', 'torch>=1.10.2',
                      'torchio>=0.18.73', 'torchvision>=0.11.3', 'matplotlib>=3.5.1', 'pandas>=1.3.5',
                      'scipy', 'tqdm', 'einops', 'python-dateutil', 'seaborn'],
    # install_requires=['nibabel==3.2.2', 'numpy==1.18.5', 'nilearn==0.9.0', 'monai==0.7.0', 'bcblib', 'torch==1.10.2',
    #                   'torchio==0.18.73', 'torchvision==0.11.3', 'matplotlib==3.5.1', 'pandas==1.3.5',
    #                   'scipy', 'tqdm', 'einops', 'python-dateutil', 'seaborn'],
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
