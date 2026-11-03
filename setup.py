from setuptools import setup, find_packages

setup(
    name="shardfit",
    version="0.1.0",
    description="Stream language model shards through VRAM to run models larger than GPU memory",
    author="ShardFit engineering team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "shardfit-profiler=shardfit.profiler:main",
            "shardfit-planner=shardfit.planner:main",
            "shardfit-chat=shardfit.cli.chat:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
