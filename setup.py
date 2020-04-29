__author__ = 'Ari-Pekka Honkanen'
__license__ = 'MIT'
__date__ = '2020-04-30'

from setuptools import setup
#
# memorandum (for pypi)
#
# python setup.py sdist upload


setup(name='pyTTE',
      version='1.0',
      description='Package to compute X-ray diffraction curves of bent crystals by numerically solving the 1D Takagi-Taupin equation.',
      author='Ari-Pekka Honkanen',
      author_email='honkanen.ap@gmail.com',
      url='https://github.com/aripekka/pyTTE/',
      packages=[
                'pyTTE',
               ],
      install_requires=[
                        'numpy>=1.16.6',
                        'scipy>=1.2.1',
                        'multiprocess>=0.70.9',
                        'matplotlib>=2.2.3'
                       ],
      include_package_data=True,
)
