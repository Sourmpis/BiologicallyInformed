from setuptools import find_packages, setup


def meta_data():
    meta = {
        "version": "0.1.0",
        "maintainer": "Guillaume Bellec and Christos Sourmpis",
        "email": "christos.sourmpis@epfl.ch; guallaume.bellec@epfl.ch",
        "url": "https://www.epfl.ch/labs/lcn/",
        "license": "Apache 2.0",
        "description": "Code for the publication Sourmpis et al 2023, Neurips in PyTorch.",
    }

    return meta


def setup_package():
    with open("README.md") as f:
        long_description = f.read()
    meta = meta_data()
    setup(
        name="infopath",
        version=meta["version"],
        description=meta["description"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Apache 2.0",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
        ],
        packages=find_packages(),
        install_requires=[
            "matplotlib==3.6.3",
            "scikit-learn==1.2.2",
            "mat73==0.58",
            "tqdm==4.64.1",
            "geomloss==0.2.5",
            "seaborn==0.12.0",
            "statannot==0.2.3",
            "torchmultitask @ git+https://github.com/guillaumeBellec/multitask",
            "jupyterlab==2.3.2",
            "jupyter",
        ],
    )


if __name__ == "__main__":
    setup_package()
