from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension
from evogression import __version__


with open('README.md', 'r') as f:
    long_description = f.read()


setup(name='evogression',
      version=__version__,
      packages=find_packages(),
      license='MIT',
      author='Zach Bateman',
      description='Evolutionary Regression',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/zachbateman/evogression.git',
      download_url='https://github.com/zachbateman/evogression/archive/v_' + __version__ + '.tar.gz',
      keywords=['REGRESSION', 'MACHINE', 'LEARNING', 'EVOLUTION'],
      install_requires=['tqdm', 'easy_multip'],
      classifiers=['Development Status :: 4 - Beta',
                    'License :: OSI Approved :: MIT License',
                    'Programming Language :: Python :: 3.9',
                    'Programming Language :: Python :: 3.10',
                    'Programming Language :: Python :: 3.11',
                    ],
    rust_extensions=[RustExtension("evogression.rust_evogression", binding=Binding.PyO3)]
)
