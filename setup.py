from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from evogression import __version__


with open('README.md', 'r') as f:
    long_description = f.read()

extensions = [
    Extension('evogression.calc_target_cython', ['evogression/calc_target_cython.pyx']),
    Extension('evogression.generate_parameter_coefficients_calc', ['evogression/generate_parameter_coefficients_calc.pyx']),
    Extension('evogression.calc_error_sum', ['evogression/calc_error_sum.pyx']),
]


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
      classifiers=['Development Status :: 3 - Alpha',
                    'License :: OSI Approved :: MIT License',
                    'Programming Language :: Python :: 3',
                    'Programming Language :: Python :: 3.7',
                    'Programming Language :: Python :: 3.8',
                    ],
    ext_modules=cythonize(extensions, compiler_directives={'language_level': '3'})
)
