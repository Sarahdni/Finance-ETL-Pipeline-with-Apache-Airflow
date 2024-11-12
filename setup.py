from setuptools import setup, find_packages

setup(
    name="finance-etl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'pytest'
    ]
)