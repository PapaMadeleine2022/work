
# make: *** [bin/im2rec] Error 1
I use ubuntu14.04, cuda8, cudnn5.1, opencv2.4.9 When I run the command

```
bash install-mxnet-ubuntu-python.sh
```
in **Quick Installation** section following the installation instructon in [url](http://mxnet.io/get_started/ubuntu_setup.html) 
, it shows error:

/usr/bin/ld: warning: libcudart.so.7.5, needed by /usr/local/lib/libopencv_core.so, not found (try using -rpath or -rpath-link)
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFReadRGBAStrip@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFIsTiled@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFWriteScanline@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFGetField@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFScanlineSize@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFReadEncodedTile@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFReadRGBATile@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFClose@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFRGBAImageOK@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFOpen@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFReadEncodedStrip@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFSetField@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFSetWarningHandler@LIBTIFF_4.0'
/usr/local/lib/libopencv_highgui.so: undefined reference to `TIFFSetErrorHandler@LIBTIFF_4.0'
collect2: error: ld returned 1 exit status
make: *** [bin/im2rec] Error 1

### [solution](https://github.com/dmlc/mxnet/issues/1135):

```
make clean_all
```

