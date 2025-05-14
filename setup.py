from setuptools import find_packages, setup

setup(
    name="s4l-fenics-plugin",
    version="1.0.0",
    description="Fenics simulation plugin for S4L",
    author="Manuel Guidon",
    package_dir={"": "src"},  # This tells setuptools to look in the src directory
    packages=find_packages(where="src"),  # This finds packages inside src
    python_requires=">=3.11",
    install_requires=[
        "s4l_core",  # Core package dependency
    ],
    entry_points={
        "s4l.simulator_plugins": [
            "fenics = fenics.register:register",
        ],
    },
)