import setuptools

# Read the contents of README.md file for long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Specify dependencies
requirements = [
    "scikit-learn",
    "keras",
    "tensorflow",
    "tensorflow-privacy",
    "keras-tuner"
]

# Define package metadata
setuptools.setup(
    name="private_targeting",
    version="0.1",
    description="A package for targeting with privacy protection by differential privacy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
)
