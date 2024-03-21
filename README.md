# Live Face Recognition

Simple implementation of live face recognition implementing LBPH algorithm for recognizing faces and HaarCascade pretrained model to detect faces.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the needed library. If you haven't installed pip, go check their website

Install numpy
```bash
pip install numpy
```

Install OpenCV
```bash
pip install opencv-contrib-python
```

Install Pillow
```bash
pip install pillow
```

## Usage

There are 3 python file in this repository
1. register-face.py
2. train.py
3. scan.py

To use this application  : 
- run the `register-face.py` to record faces that you want to recognize
- the recorded face data will be saved under the data directory
- after recording faces, run the `training.py` to train the recognizer and the output will be a file named `training.xml`
- run the `scan.py` to implement the `training.xml` and enjoy a simple live face recognition on your PC

## Contributing
I know my codes are messy and ineffective, so I am always happy to accept your pull request to fix my messy code.


Please make sure to update tests as appropriate.
