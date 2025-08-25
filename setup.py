#!/usr/bin/env python3
"""
Setup script for Kurachi AI Multilingual Pipeline
"""
from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README_MULTILINGUAL.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Kurachi AI - Multilingual Conversation Interface"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="kurachi-multilingual",
    version="1.0.0",
    description="Multilingual conversation interface with intelligent translation and cultural adaptation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Kurachi AI Team",
    author_email="dev@kurachi.ai",
    url="https://github.com/kurachi-ai/multilingual-pipeline",
    
    packages=find_packages(include=['services', 'services.*', 'models', 'models.*', 'utils', 'utils.*']),
    include_package_data=True,
    
    install_requires=read_requirements(),
    
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'pytest-timeout>=2.1.0',
        ]
    },
    
    python_requires=">=3.8",
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    entry_points={
        'console_scripts': [
            'kurachi-preferences=cli_preferences:main',
            'kurachi-setup=scripts.setup_multilingual:main',
        ],
    },
    
    package_data={
        'config': ['*.yml', '*.yaml', '*.json'],
        'docs': ['*.md'],
    },
    
    zip_safe=False,
)