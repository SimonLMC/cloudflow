import sys

from pkg_resources import VersionConflict, require
from setuptools import setup
from pathlib import Path

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


def read_requirements(basename):
    requirements_file = Path(__file__).parent.absolute() / basename
    with open(requirements_file) as f:
        return [req.strip() for req in f.readlines()]


if __name__ == "__main__":
    requirements = read_requirements("requirements.txt")
    setup(
        use_pyscaffold=True,
        install_requires=requirements
    )