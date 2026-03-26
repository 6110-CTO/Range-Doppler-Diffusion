from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rddiffusion",
    version="0.1.0",
    author="Ari Gra",
    author_email="",  # Add your email if you want
    description="Diffusion model for radar target detection using UNet architecture with Student-T noise scheduling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arigra/rddiff",
    package_dir={"rddiffusion": "src"},
    packages=["rddiffusion", "rddiffusion.models"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rddiffusion=main:main",
        ],
    },
)
