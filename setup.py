from setuptools import find_packages, setup
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="dsipts",
    version="1.1.0",
    author="Andrea Gobbi",
    author_email="agobbi@fbk.eu",
    packages=find_packages(exclude=("tests",)),
    description="Python library for time series forecasting",
    setup_requires=[],
    install_requires=requirements,
)

'''
"""Custom clean command to tidy up the project root."""
CLEAN_FILES = ['build', 'dist', 'egg-info']


here = os.getcwd()

for dir in os.listdir(here):
    if any( [f in dir for f in CLEAN_FILES] ):
        shutil.rmtree(os.path.join(here,dir))
'''