The C++ implementation based on Libtorch is shown in this dir.

### installation ###
Recently the project is tested on win10. Only supported on Libtorch-release.
Dependency as bellow:
```commandline
CUDA 11.7 
cudnn 8.5.0
VC++ 14
VS 2019 & VS 2019 buildtools
cmake 4.2.1
gcc 15.2.0(install from msys2, Mingw64)
opencv 3.4.9
libtorch-win-shared-with-deps-1.13.1%2Bcu117
```

### build ###
run the command below
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```
Then you can find `inference_app.vcxproj` in dir "./build", which can be open by Visual Studio.

Because of some unknown reasons, some .dll files cannot be located directly, which contain in the folder `dll_you_need_release`.
Follow the steps below to fix it.

1. Debug or Run the project in VS with the mode `release x64`.
2. Find the folder `./build/release` just created, with the executable file inference_app.exe .
3. copy the dll files in `dll_you_need_release` to the folder `./build/release`
> A more "officially" but not correct either is find the dll files in Libtorch path(such as `libtorch-win-shared-with-deps-debug-1.13.1%2Bcu117/libtorch/lib`) and OpenCV Path(such as `opencv/build/x64/vc14/bin`, and notice the vc++ version)

 ### Run ###
Now `inference_app.exe` (in folder release) can run correctly, after running the result can be found in current dir.

### Notice ###
The torch.jit model is loaded with trace mode, and to make the stability of the model, the input image is resized to (512, 512) before fed into the model.
