from setuptools import setup, Extension

module = Extension('PATTACK',
                   sources=['main_q.cpp'],
                   include_dirs=['/usr/local/include'],
                   libraries=['m'],
                   library_dirs=['/usr/local/lib'])

setup(name='PATTACK',
      version='1.0',
      description='Python interface for custom C code',
      ext_modules=[module])
