from setuptools import setup, find_packages

setup(
    name='swinir_pipeline',
    version='0.1.0',
    description='A brief description of your project',
    author='Aleksandr Razin',
    author_email='razin.x.aleks@google.com',
    url='pass',
    packages=find_packages(),
    install_requires=[
        'torch',
        #'torchvision',
        #'PILLOW',
        #'diffusers',
        #'transformers',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='pytorch dataset images depth_map',
)
