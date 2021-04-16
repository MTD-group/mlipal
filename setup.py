import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlipal", # Replace with your own username
    version="0.0.1",
    author="Sean Koyama",
    author_email="koysean@gmail.com",
    description="Package to create machine-learned interatomic potentials using\
            active learning to facilitate efficient model training.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
