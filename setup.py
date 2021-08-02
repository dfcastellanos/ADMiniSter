import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ADMiniSter",
    version="0.3",
    author="David Fernández Castellanos",
    author_email="castellanos@gmx.com",
    description="A minimalist suite for managing plain text files storing numerical data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kastellane/ADMiniSter",
    project_urls={
        "Bug Tracker": "https://github.com/kastellane/ADMiniSter/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.4",
    install_requires=['joblib','progressbar2','numpy','pandas']
)