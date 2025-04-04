from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="trading",
    version="2.0.0",
    author="Calvin Bergin",
    author_email="calvin@app-support.com",
    description="A sophisticated trading system with multiple strategies, machine learning capabilities, and state-of-the-art anomaly detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Calvindd2f/trading",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": [
            "torch>=1.10.0",
            "torchvision>=0.11.0",
            "torchaudio>=0.10.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0.0"
        ],
        "anomaly": [
            "scikit-learn>=1.0.0",
            "torch>=1.10.0",
            "ta-lib>=0.4.24",
            "pandas-ta>=0.3.14b",
            "optuna>=2.10.0"
        ],
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.9b0",
            "flake8>=3.9.2",
            "isort>=5.9.3",
            "mypy>=0.910"
        ]
    },
    entry_points={
        "console_scripts": [
            "trading-train=src.blueprints.training:main",
            "trading-web=src.web.app:main",
            "trading-anomaly=src.core.strategies.anomaly_detection:main"
        ],
    },
) 