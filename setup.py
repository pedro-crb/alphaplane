from setuptools import setup, find_packages

setup(
    name='alphaplane',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        'alphaplane': ['*.exe', '*.json', '*.dat', '*.PE0'],
    },
    include_package_data=True,
    install_requires=[
        'dill>=0.3.8',
        'matplotlib>=3.9.2',
        'numpy==1.26.4',
        'pandas>=2.2.2',
        'pyvista>=0.44.1',
        'scipy>=1.14.1',
    ],
    author='Pedro Ribeiro',
    author_email='pedro.c.rib@usp.br',
    description='alphaplane is an aircraft analysis and design library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pedro-crb/alphaplane_release',
)