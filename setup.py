from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="cellnavi",
    version="0.0.1",
    description="CellNavi is a deep learning framework designed to predict genes driving cellular transitions."
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.12",
)