import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tfrecordutils",
    version="0.1.3",
    description="Utilities to simplify working with TFRecord files and TensorFlow's data pipeline API",
    long_description=long_description,
    author="Dave MacDonald",
    author_email="dave@torontoai.org",

    packages=["tfrecordutils"],
    include_package_data=True,

    # Details
    url="https://github.com/mindlapse/tfrecordutils",

    # Dependent packages (distributions)
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]

)
