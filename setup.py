from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this returns list of requirements from a file
    
    '''
    ee='-e .'
    req=[]
    with open(file_path,'r') as f:
        req=f.read().splitlines()
    if ee in req:
        req.remove(ee)
    return req

setup(
    name='mymlproject',
    version='0.0.1',
    author='RAM',
    author_email='ramk612000@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('req.txt'),
)