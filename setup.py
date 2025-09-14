#!/usr/bin/env python3
“””
ITPU (Information-Theoretic Processing Unit) Setup
“””

from setuptools import setup, find_packages
import os

# Read the README file

def read_readme():
readme_path = os.path.join(os.path.dirname(**file**), ‘README.md’)
try:
with open(readme_path, ‘r’, encoding=‘utf-8’) as f:
return f.read()
except FileNotFoundError:
return “ITPU: Information-Theoretic Processing Unit”

# Read requirements

def read_requirements(filename=‘requirements.txt’):
req_path = os.path.join(os.path.dirname(**file**), filename)
try:
with open(req_path, ‘r’, encoding=‘utf-8’) as f:
# Filter out comments and empty lines
requirements = []
for line in f:
line = line.strip()
if line and not line.startswith(’#’):
requirements.append(line)
return requirements
except FileNotFoundError:
return [
‘numpy>=1.20.0’,
‘scipy>=1.7.0’,
‘matplotlib>=3.5.0’
]

# Core requirements (always needed)

core_requirements = [
‘numpy>=1.20.0’,
‘scipy>=1.7.0’,
‘matplotlib>=3.5.0’,
]

# Optional requirements for different use cases

extras_require = {
‘performance’: [
‘numba>=0.56.0’,
‘cupy>=10.0.0’,  # GPU support
],
‘neuroscience’: [
‘mne>=1.0.0’,
‘nibabel>=3.2.0’,
‘scikit-learn>=1.0.0’,
],
‘medical’: [
‘nibabel>=3.2.0’,
‘pydicom>=2.0.0’,
‘opencv-python>=4.5.0’,
],
‘benchmarks’: [
‘scikit-learn>=1.0.0’,
‘pandas>=1.3.0’,
‘seaborn>=0.11.0’,
],
‘dev’: [
‘pytest>=6.0.0’,
‘pytest-cov>=3.0.0’,
‘black>=22.0.0’,
‘flake8>=4.0.0’,
‘mypy>=0.950’,
‘sphinx>=4.0.0’,
‘sphinx-rtd-theme>=1.0.0’,
],
‘demos’: [
‘notebook>=6.4.0’,
‘ipywidgets>=7.6.0’,
‘plotly>=5.0.0’,
‘streamlit>=1.0.0’,
],
‘all’: [
‘numba>=0.56.0’,
‘mne>=1.0.0’,
‘nibabel>=3.2.0’,
‘scikit-learn>=1.0.0’,
‘pandas>=1.3.0’,
‘seaborn>=0.11.0’,
‘notebook>=6.4.0’,
‘ipywidgets>=7.6.0’,
‘plotly>=5.0.0’,
]
}

setup(
name=“itpu”,
version=“0.1.0-alpha”,
author=“Justin Bilyeu”,
author_email=“justin@yourdomain.com”,  # Update with actual email
description=“Information-Theoretic Processing Unit: Fast entropy, mutual information, and k-NN statistics”,
long_description=read_readme(),
long_description_content_type=“text/markdown”,
url=“https://github.com/justindbilyeu/ITPU”,
project_urls={
“Bug Tracker”: “https://github.com/justindbilyeu/ITPU/issues”,
“Documentation”: “https://github.com/justindbilyeu/ITPU/docs”,
“Source Code”: “https://github.com/justindbilyeu/ITPU”,
},
packages=find_packages(where=“src”),
package_dir={””: “src”},
classifiers=[
“Development Status :: 3 - Alpha”,
“Intended Audience :: Science/Research”,
“Intended Audience :: Developers”,
“License :: OSI Approved :: Apache Software License”,
“Operating System :: OS Independent”,
“Programming Language :: Python :: 3”,
“Programming Language :: Python :: 3.8”,
“Programming Language :: Python :: 3.9”,
“Programming Language :: Python :: 3.10”,
“Programming Language :: Python :: 3.11”,
“Programming Language :: Python :: 3.12”,
“Topic :: Scientific/Engineering”,
“Topic :: Scientific/Engineering :: Information Analysis”,
“Topic :: Scientific/Engineering :: Medical Science Apps.”,
“Topic :: Software Development :: Libraries :: Python Modules”,
],
python_requires=”>=3.8”,
install_requires=core_requirements,
extras_require=extras_require,
include_package_data=True,
package_data={
“itpu”: [“py.typed”],  # Mark as typed package
},
entry_points={
“console_scripts”: [
“itpu-benchmark=itpu.cli:benchmark_cli”,
“itpu-demo=itpu.cli:demo_cli”,
],
},
keywords=[
“mutual information”,
“entropy”,
“information theory”,
“neuroscience”,
“BCI”,
“medical imaging”,
“signal processing”,
“streaming”,
“real-time”,
“FPGA”,
“hardware acceleration”
],
zip_safe=False,
)
