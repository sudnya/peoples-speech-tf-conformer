from setuptools import setup, find_packages

setup(
    name='PeoplesSpeechTensorflowConformer',
    version='0.1dev',
    packages=find_packages(),
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    scripts=['peoples_speech_tf_conformer/run-peoples-speech.py'],
    install_requires=[
        'TensorFlowASR[tf2.x]',
        'tensorflow~=2.6.0',
        'tensorflow-text~=2.6.0',
        'tensorflow-io~=0.20.0'
        
    ]
)

