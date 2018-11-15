from setuptools import setup, find_packages
import subprocess

def get_version():
    p = subprocess.run("git describe | grep -o -E \"v[0-9]+(\\.[0-9]+)+(-[0-9]+)?\" -", shell=True, check=True,
                       universal_newlines=True, stdout=subprocess.PIPE)
    v = p.stdout.rstrip()
    return v.replace("-", ".dev", 1).replace("v", "", 1)

version = get_version()
print("Package version: " + version)

setup(
    name='eigenio',

    version=version,
    
    description='The EigenIO python library',
    #long_description="",

    url='https://github.com/crey0/eigenio',

    author='Christophe Reymann',
    author_email='',
    
    license='MPL-2.0',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.5',
    ],
    package_dir={'':'python'},
    packages=find_packages(where='python'),
    install_requires = [],
    
)
