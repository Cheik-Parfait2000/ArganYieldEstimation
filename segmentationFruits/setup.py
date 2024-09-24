from setuptools import setup, find_packages
import pathlib


# Get the parent directory
here = pathlib.Path(__file__).parent.resolve()

setup(
    name="fruits-segmentor",  # Required. Name of the package to be used for installation : pip install name
    version="0.0.1",  # Required. version of the package
    description="A project for argan fruits counting from images",
    keywords="Fruits counting, Deep Learning",
    # package_dir={"": "src"} : specify if the package is in a subdirectory. e.g src/mypackage
    packages=find_packages(include=["FruitsSegmentor"]),  # Find all the packages of my project
    python_requires=">=3.8, <4",
    # list of packages that the module depends on. Will also be installed with the package
    install_requires=["torch", "torchvision", "scikit-image", "segmentation-models-pytorch", "numpy",
                      "pandas", "geopandas", "albumentations", "opencv-contrib-python", "matplotlib",
                      "patched_yolo_infer", "ultralytics", "pickle"]
)
