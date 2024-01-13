from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file=file_path, mode='r') as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.replace('\n', '') for req in requirements]
            if "-e ." in requirements:
                  requirements.remove("-e .")
    return requirements


setup(
    name="End To End Machine Learning Project", 
    version='0.1',
    author="Nouri-Muhammad",
    author_email="nouri.muhammad1991@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
