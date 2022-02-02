import setuptools

install_requires = [
        'tensorflow-gpu==1.14.0',
        'gym',
        'gym[atari]',
        'tensorboardX',
]

setuptools.setup(
    name="soda",
    version='0.1',
    install_requires=install_requires,
    packages=setuptools.find_packages(),
)

