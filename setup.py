from setuptools import find_packages, setup


def get_version(path):
    for line in open(path):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError(f"Unable to find __version__ in {path}")


def get_requirements(path):
    res = []
    for line in open(path):
        line = line.split('#')[0].strip()
        if line:
            res.append(line)
    return res


setup(
    name='cerberus',
    version=get_version('cerberus/__init__.py'),
    author='Sean MacAvaney',
    author_email='sean.macavaney@glasgow.ac.uk',
    description="Cerberus",
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/seanmacavaney/cerberus',
    packages=find_packages(),
    include_package_data=True,
    entry_points={},
    install_requires=get_requirements('requirements.txt'),
    python_requires='>=3.10',
)
