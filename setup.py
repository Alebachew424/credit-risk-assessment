from setuptools import setup, find_packages

setup(
    name="credit-risk-assessment",
    version="1.0.0",
    author="Your Name",
    description="Machine learning model to predict credit default risk",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "imbalanced-learn>=0.10.0",
    ],
)
