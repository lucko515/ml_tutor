from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = ["matplotlib>=3.2.1",
                    "pandas>=0.25.1",
                    "numpy>=1.18.1",
                    "scikit_learn>=0.23.2"
                    ]

setup(name='ml_tutor',
      version='1.0.0.1',
      description='ML Tutor : Learn Machine Learning while never leaving the conform of your Python IDE (Jupyter Notebook or Google Colab)',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/lucko515/ml_tutor',
      author='Luka Anicin',
      author_email='luka.anicin@gmail.com',
      license='MIT',
      install_requires=install_requires,
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
