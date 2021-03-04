from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='lesseg_unet',        # This is the name of your PyPI-package.
    version='0.2',     # Update the version number for new releases
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(exclude=['__pycache__']),
    install_requires=['nibabel>=3', 'numpy', 'nilearn', 'monai', 'bcblib', 'torch', 'torchio',
                      'torchvision', 'matplotlib'],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst", "*.md"],
        # Include all the executables from bin
        "data": ["*"],
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
