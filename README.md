# distil-image-upscale

This repository contains the source code to compile an image upscale dynamic library for upscaling images. The library upscales the supplied image by a factor of 2 using machine learning models. This library should compile on Linux, MacOS, and Windows. Note: GPU support is only on Linux and Windows.

# Dependencies

> Tensorflow c
> https://www.tensorflow.org/install/lang_c

# Installation

Make sure Tensorflow c is installed on your machine then link to that library when compiling (similar to below).

```console
gcc -I/usr/local/include -L/usr/local/lib main.c -ltensorflow -o main
```

# Credits

This repository is a c binding of the work image-super-resolution.
https://github.com/idealo/image-super-resolution
