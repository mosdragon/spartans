FROM ubuntu:14.04

# Ubuntu sides with libav, I side with ffmpeg.
# RUN echo "deb http://ppa.launchpad.net/jon-severinsson/ffmpeg/ubuntu quantal main" >> /etc/apt/sources.list
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1DB8ADC1CFCA9579


RUN apt-get update
RUN apt-get install -y -q wget curl zip
RUN apt-get install -y -q build-essential
RUN apt-get install -y -q cmake
RUN apt-get install -y -q python2.7 python2.7-dev
RUN wget 'https://pypi.python.org/packages/2.7/s/setuptools/setuptools-0.6c11-py2.7.egg' && /bin/sh setuptools-0.6c11-py2.7.egg && rm -f setuptools-0.6c11-py2.7.egg
RUN curl 'https://bootstrap.pypa.io/get-pip.py' | python2.7
RUN pip install numpy
ADD build_opencv.sh /build_opencv.sh
RUN /bin/sh /build_opencv.sh
RUN rm -rf /build_opencv.sh

RUN apt-get install -y python-skimage
RUN pip install flask

RUN apt-get install -y git libboost-all-dev
RUN git clone 'https://github.com/davisking/dlib' dlib
RUN python2.7 dlib/setup.py install
RUN rm -rf dlib