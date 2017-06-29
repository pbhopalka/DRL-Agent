sudo pip install --upgrade pip
sudo pip install --upgrade tensorflow
sudo pip install gym
sudo apt-get install libjpeg-turbo8-dev cmake make golang zlib1g-dev
sudo pip install gym[atari]
sudo pip install opencv-python
sudo git clone https://github.com/openai/universe.git
cd universe
sudo pip install -e .
cd ..
sudo rm -r universe
