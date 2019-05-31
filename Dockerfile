FROM nvcr.io/nvidia/tensorflow:18.11-py3
RUN pip install --upgrade pip
RUN pip install pandas bcolz h5py matplotlib mkl nose jupyter Pillow pandas pydot pyyaml scikit-learn six keras tensorboardX
