from setuptools import setup, find_packages

setup(
    name="elegantrl_dq",
    version="0.3.2",
    author="Kideng",
    author_email="luo_li_ba_suo@163.com",
    url="https://github.com/luo-li-ba-suo/Robomaster/",
    license="Apache 2.0",
    packages=['elegantrl_dq'],
    install_requires=[
        'gym', 'matplotlib', 'numpy'],
    description="Lightweight, Efficient and Stable DRL Implementation Using PyTorch",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords="Deep Reinforcment Learning",
    python_requires=">=3.6",
)
