import os
from setuptools import setup, findpackages

# Export PYTHONPATH=$PYTHONPATH:/C:/It's me/Apun/Deep Learning/CV/Self Driving Car

# Load requirements from requirements.txt
with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

# Create necessary folders if they don't exist
dirs=[
    'src/models',
    'src/inference',
    'data',
    'saved_models/regression_model',
    'saved_models/lane_segmentation_model',
    'saved_models/object_detection_model',
    'utils',
    'notebooks',
    'tests'
]

for dir in dirs:
    os.makedirs(dir, exist_ok=True)

setup(
    name='Self Driving Car',
    version='0.3',
    #version='0.0.1',
    packages=findpackages(where='src'),
    package_dir={'': 'src'},
    install_requires=required_packages,
    entry_points={
        'console_scripts': [
            'run_fsd_inference=src.inference.run_fsd_inference:main',
            'run_segmentation=src.inference.run_segmentation:main',
            'run_steering=src.inference.run_steering_angle_prediction:main'
        ]
    },
    author='<Ashutosh Singh',
    description='A Self Driving Car project using Computer Vision and Deep Learning',
    license='MIT',
    classifiers=[
        'Development Status :: 4-Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
)