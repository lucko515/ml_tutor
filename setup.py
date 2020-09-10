from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='ml_tutor',
      version='0.9.2',
      description='ML Tutor is a Python library that helps you to learn Machine Learning.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/lucko515/ml_tutor',
      author='Luka Anicin',
      author_email='luka.anicin@gmail.com',
      license='MIT',
      packages=find_packages(),
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Intended Audience :: Education",
            "Intended Audience :: Developers",
            "Intended Audience :: Information Technology",
            "Intended Audience :: Science/Research"
      ],
      python_requires='>=3',
      )
