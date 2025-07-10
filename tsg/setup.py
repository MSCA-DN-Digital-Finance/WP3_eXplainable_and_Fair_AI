from setuptools import setup, find_packages

setup(
    name='tsg',
    version='0.1.0',
    description='Modular Time Series Generator Library',
    authors=[
        'Mathis Jander',
        'Jens Reil'
    ],
    authors_email=['mathis.jander@utwente.nl', 'jens.reil@bfh.ch'],
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21'
    ],
    python_requires='>=3.8',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
