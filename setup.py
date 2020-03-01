import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


with open('README.md', 'r') as f:
    long_description = f.read()

# extensions = [
    # Extension('evogression.single_layer_calc', ['evogression/single_layer_calc.pyx']),
    # Extension('evogression.generate_parameter_coefficients_calc', ['evogression.generate_parameter_coefficients_calc.pyx']),
# ]


setup(name='evogression',
      version='0.2.0',
      packages=['evogression'],
      license='MIT',
      author='Zach Bateman',
      description='Evolutionary Regression',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/zachbateman/evogression.git',
      download_url='https://github.com/zachbateman/evogression/archive/v_0.2.0.tar.gz',
      keywords=['REGRESSION', 'MACHINE', 'LEARNING', 'EVOLUTION'],
      install_requires=['tqdm', 'easy_multip'],
      classifiers=['Development Status :: 3 - Alpha',
                    'License :: OSI Approved :: MIT License',
                    'Programming Language :: Python :: 3',
                    'Programming Language :: Python :: 3.6',
                    'Programming Language :: Python :: 3.7',
                    'Programming Language :: Python :: 3.8',
                    ],
    # ext_modules=cythonize(extensions)
    ext_modules=cythonize(['evogression\\*.pyx'])
)
