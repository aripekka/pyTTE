__author__ = 'Ari-Pekka Honkanen'
__license__ = 'MIT'
__date__ = '17/11/2017'

from setuptools import setup
#
# memorandum (for pypi)
#
# python setup.py sdist upload


setup(name='pyTTE',
      version='0.1.8',
      description='Package to compute X-ray diffraction curves of bent crystals by numerically solving the Takagi-Taupin equations.',
      author='Ari-Pekka Honkanen',
      author_email='honkanen.ap@gmail.com',
      url='https://github.com/aripekka/pyTTE/',
      packages=[
                'pyTTE',
                'pyTTE.examples',
               ],
      install_requires=[
                        'numpy',
                        'scipy',
                        'matplotlib'
                       ],
      include_package_data=True,
)
