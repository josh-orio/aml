## cuBLAS Installation

Installing cuBLAS is pretty easy these days!

You will need the CUDA toolkit, which ships with a lot of things, including cuBLAS.

Your installation commands will look something like this:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```

This works for Ubuntu 24.04, but depending on your distro, OS version and some other things, you may want to do a search for "cuda toolkit install {distro}" to get the correct version.
