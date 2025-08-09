from setuptools import find_packages, setup

setup(
    name="omni_drones",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        # "hydra-core>=1.3.2",
        # "omegaconf>=2.3.0",
        # "wandb==0.12.21",
        # "moviepy>=1.0.3",
        # "imageio>=2.31.1",
        # "plotly>=5.18.0",
        # "einops>=0.7.0",
        # "av>=10.0.0",  # for moviepy
        # "pandas>=2.1.0",
        # install by cloning from github
        # "tensordict" 
        # "torchrl",
    ],
    python_requires='>=3.8,<=3.11',
)
