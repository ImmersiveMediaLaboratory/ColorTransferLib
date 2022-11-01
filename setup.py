import setuptools

setuptools.setup(
    name="colortransfer",
    version="0.0.2",
    author="Herbert Potechius",
    author_email="potechius.herbert@gmail.com",
    description="A small example package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={"": ['Options/*.json','Algorithms/TpsColorTransfer/L2RegistrationForCT/*/*/*']},
    include_package_data=True,
    python_requires='>=3.8',
)