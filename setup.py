from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize


with open('README.md', 'r') as f:
    long_description = f.read()

extensions = [
    Extension('evogression.calc_target_cython', ['evogression/calc_target_cython.pyx']),
    Extension('evogression.generate_parameter_coefficients_calc', ['evogression/generate_parameter_coefficients_calc.pyx']),
]


setup(name='evogression',
      version='0.3.1',
      packages=find_packages(),
      license='MIT',
      author='Zach Bateman',
      description='Evolutionary Regression',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/zachbateman/evogression.git',
      download_url='https://github.com/zachbateman/evogression/archive/v_0.3.1.tar.gz',
      keywords=['REGRESSION', 'MACHINE', 'LEARNING', 'EVOLUTION'],
      install_requires=['tqdm', 'easy_multip'],
      classifiers=['Development Status :: 3 - Alpha',
                    'License :: OSI Approved :: MIT License',
                    'Programming Language :: Python :: 3',
                    'Programming Language :: Python :: 3.6',
                    'Programming Language :: Python :: 3.7',
                    'Programming Language :: Python :: 3.8',
                    ],
    ext_modules=cythonize(extensions)
)
