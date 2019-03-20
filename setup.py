import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fml",
    version="0.1.0",
    author="Francis Williams",
    author_email="francis@fwilliams.info",
    description=" FML (Francis' Machine-Learnin' Library) - A collection of utilities for machine learning tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fwilliams/fml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
