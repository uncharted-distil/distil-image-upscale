# distil-image-upscale

This repository contains the source code to compile an image upscale dynamic library for upscaling images. The library upscales the supplied image by a factor of 2 using machine learning models. This library should compile on Linux, MacOS, and Windows. Note: GPU support is only on Linux and Windows.

# Binaries
<ul>
  <li><a href="https://github.com/uncharted-distil/distil-image-upscale/releases/download/1.1-linux-cpu/image-upscale.so">Linux CPU</a></li>  
  <li><a href="https://github.com/uncharted-distil/distil-image-upscale/releases/download/1.1-linux-gpu/image-upscale.so">Linux GPU</a></li>  
  <li><a href="https://github.com/uncharted-distil/distil-image-upscale/releases/download/1.1-mac-cpu/image-upscale.so">macOS CPU</a></li> 
</ul>

# Dependencies

> Tensorflow c
> https://www.tensorflow.org/install/lang_c

# Installation

Make sure Tensorflow c is installed on your machine then link to that library when compiling (similar to below).

```console
gcc -I /usr/local/include/ -L /usr/local/lib/ -Wall -fPIC -c src/entry_functions.c -ltensorflow -o entry_functions.o
gcc -shared -o image-upscale.so entry_functions.o
```
# Notes
> The data supplied through the InputData.data void pointer MUST BE CONTIGUOUS MEMORY. Arrays of pointers will not ingest properly.
# Credits

This repository is a c binding of the work image-super-resolution.
https://github.com/idealo/image-super-resolution
