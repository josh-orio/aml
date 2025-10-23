## OpenBLAS Installation

Installing OpenBLAS is pretty easy!

OpenBLAS needs to be built from source and then installed locally. The commands will look like:

```bash
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
mkdir build && cd build
cmake .. && sudo make -j install
```

Some systems like a raspi4 cannot build OpenBLAS with the -j flag, but on capable systems it speeds up the compilation a lot.
