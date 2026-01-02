"""
Setup script for MAPPO-ABC
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mappo-abc-smac",
    version="1.0.0",
    author="Koray Özdemir, Muhammed Şara, Adem Tuncer, Süleyman Eken",
    author_email="adem.tuncer@yalova.edu.tr",
    description="Multi-Agent PPO with Artificial Bee Colony Optimization for SMAC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kryozdemir/MAPPO-ABC",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
    },
)
