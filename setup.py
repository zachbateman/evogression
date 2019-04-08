import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


with open('README.md', 'r') as f:
    long_description = f.read()

extensions = [
    Extension('evogression.single_layer_calc', ['evogression/single_layer_calc.pyx'])
]


setup(name='evogression',
      version='0.1.0',
      packages=['evogression'],
      # license='',
      author='Zach Bateman',
      description='Evolutionary Derived Regression',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/zachbateman/evogression.git',
      download_url='https://github.com/zachbateman/evogression/archive/v_0.1.0.tar.gz',
      keywords=['REGRESSION', 'MACHINE', 'LEARNING', 'EVOLUTION'],
      install_requires=['tqdm'],
      classifiers=['Development Status :: 3 - Alpha',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   ],
    ext_modules=cythonize(extensions)
)
