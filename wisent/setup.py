from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wisent",
    version="0.1.1",
    author="Wisent Team",
    author_email="info@wisent.ai",
    description="Client library for interacting with the Wisent backend services",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wisent-ai/wisent",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "pydantic>=2.0.0",
        "aiohttp>=3.8.0",
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
        "transformers>=4.30.0",
    ],
)
