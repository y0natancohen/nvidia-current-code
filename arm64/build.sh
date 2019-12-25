echo "compiling..."
echo "sudo /usr/bin/aarch64-linux-gnu-g++     -I/usr/aarch64-linux-gnu/include -O2 -Wall -W -fPIC -D_REENTRANT -D_GNU_SOURCE -D_MAJOR_VERSION=2 -D_MINOR_VERSION=35 -D_BUILD_VERSION=0 -D_BUILD2_VERSION=2788 -pedantic -DMALLOC_TRACE -std=c++11 -I/usr/local/include -I/usr/local/include/opencv -L/usr/local/lib -L../../lib/arm64 -lopencv_highgui -lopencv_imgproc -lopencv_core -lstdc++ -lrt   -DNDEBUG -fvisibility=hidden    -I../../..    -c -o CaptureToUserMemory.legacy.o ../CaptureToUserMemory.legacy.cpp"

sudo /usr/bin/aarch64-linux-gnu-g++     -I/usr/aarch64-linux-gnu/include -O2 -Wall -W -fPIC -D_REENTRANT -D_GNU_SOURCE -D_MAJOR_VERSION=2 -D_MINOR_VERSION=35 -D_BUILD_VERSION=0 -D_BUILD2_VERSION=2788 -pedantic -DMALLOC_TRACE -std=c++11 -I/usr/local/include -I/usr/local/include/opencv -L/usr/local/lib -L../../lib/arm64 -lopencv_highgui -lopencv_imgproc -lopencv_core -lstdc++ -lrt   -DNDEBUG -fvisibility=hidden    -I../../..    -c -o CaptureToUserMemory.legacy.o ../CaptureToUserMemory.legacy.cpp

echo "linking..."
echo "sudo /usr/bin/aarch64-linux-gnu-g++ -o CaptureToUserMemory CaptureToUserMemory.legacy.o CPreprocessIm.cpp.o CPostProcChaudhuri.cpp.o CIniReader.cpp.o CImProcAlgo.cpp.o CCaudhuri.cpp.o  -Wl,-unresolved-symbols=ignore-in-shared-libs -L../../../lib/arm64 -lmvDeviceManager -lmvPropHandling      -lm -lpthread     -I/usr/aarch64-linux-gnu/include -O2 -Wall -W -fPIC -D_REENTRANT -D_GNU_SOURCE -D_MAJOR_VERSION=2 -D_MINOR_VERSION=35 -D_BUILD_VERSION=0 -D_BUILD2_VERSION=2788 -pedantic -DMALLOC_TRACE -std=c++11 -I/usr/local/include -I/usr/local/include/opencv -L/usr/local/lib -L../../lib/arm64 -lopencv_highgui -lopencv_imgproc -lopencv_core -lstdc++ -lrt   -DNDEBUG -fvisibility=hidden    -I../../..   -ldl -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64:/usr/local/lib /usr/local/lib/libopencv_gapi.so.4.1.0 /usr/local/lib/libopencv_stitching.so.4.1.0 /usr/local/lib/libopencv_aruco.so.4.1.0 /usr/local/lib/libopencv_bgsegm.so.4.1.0 /usr/local/lib/libopencv_bioinspired.so.4.1.0 /usr/local/lib/libopencv_ccalib.so.4.1.0 /usr/local/lib/libopencv_cudabgsegm.so.4.1.0 /usr/local/lib/libopencv_cudafeatures2d.so.4.1.0 /usr/local/lib/libopencv_cudaobjdetect.so.4.1.0 /usr/local/lib/libopencv_cudastereo.so.4.1.0 /usr/local/lib/libopencv_dnn_objdetect.so.4.1.0 /usr/local/lib/libopencv_dpm.so.4.1.0 /usr/local/lib/libopencv_face.so.4.1.0 /usr/local/lib/libopencv_freetype.so.4.1.0 /usr/local/lib/libopencv_fuzzy.so.4.1.0 /usr/local/lib/libopencv_hfs.so.4.1.0 /usr/local/lib/libopencv_img_hash.so.4.1.0 /usr/local/lib/libopencv_line_descriptor.so.4.1.0 /usr/local/lib/libopencv_quality.so.4.1.0 /usr/local/lib/libopencv_reg.so.4.1.0 /usr/local/lib/libopencv_rgbd.so.4.1.0 /usr/local/lib/libopencv_saliency.so.4.1.0 /usr/local/lib/libopencv_stereo.so.4.1.0 /usr/local/lib/libopencv_structured_light.so.4.1.0 /usr/local/lib/libopencv_superres.so.4.1.0 /usr/local/lib/libopencv_surface_matching.so.4.1.0 /usr/local/lib/libopencv_tracking.so.4.1.0 /usr/local/lib/libopencv_videostab.so.4.1.0 /usr/local/lib/libopencv_xfeatures2d.so.4.1.0 /usr/local/lib/libopencv_xobjdetect.so.4.1.0 /usr/local/lib/libopencv_xphoto.so.4.1.0 /usr/local/lib/libopencv_shape.so.4.1.0 /usr/local/lib/libopencv_datasets.so.4.1.0 /usr/local/lib/libopencv_plot.so.4.1.0 /usr/local/lib/libopencv_text.so.4.1.0 /usr/local/lib/libopencv_dnn.so.4.1.0 /usr/local/lib/libopencv_ml.so.4.1.0 /usr/local/lib/libopencv_phase_unwrapping.so.4.1.0 /usr/local/lib/libopencv_cudacodec.so.4.1.0 /usr/local/lib/libopencv_cudaoptflow.so.4.1.0 /usr/local/lib/libopencv_cudalegacy.so.4.1.0 /usr/local/lib/libopencv_cudawarping.so.4.1.0 /usr/local/lib/libopencv_optflow.so.4.1.0 /usr/local/lib/libopencv_video.so.4.1.0 /usr/local/lib/libopencv_ximgproc.so.4.1.0 /usr/local/lib/libopencv_objdetect.so.4.1.0 /usr/local/lib/libopencv_calib3d.so.4.1.0 /usr/local/lib/libopencv_features2d.so.4.1.0 /usr/local/lib/libopencv_flann.so.4.1.0 /usr/local/lib/libopencv_highgui.so.4.1.0 /usr/local/lib/libopencv_videoio.so.4.1.0 /usr/local/lib/libopencv_imgcodecs.so.4.1.0 /usr/local/lib/libopencv_photo.so.4.1.0 /usr/local/lib/libopencv_cudaimgproc.so.4.1.0 /usr/local/lib/libopencv_cudafilters.so.4.1.0 /usr/local/lib/libopencv_imgproc.so.4.1.0 /usr/local/lib/libopencv_cudaarithm.so.4.1.0 /usr/local/lib/libopencv_core.so.4.1.0 /usr/local/lib/libopencv_cudev.so.4.1.0"

sudo /usr/bin/aarch64-linux-gnu-g++ -o CaptureToUserMemory CaptureToUserMemory.legacy.o CPreprocessIm.cpp.o CPostProcChaudhuri.cpp.o CIniReader.cpp.o CImProcAlgo.cpp.o CCaudhuri.cpp.o  -Wl,-unresolved-symbols=ignore-in-shared-libs -L../../../lib/arm64 -lmvDeviceManager -lmvPropHandling      -lm -lpthread     -I/usr/aarch64-linux-gnu/include -O2 -Wall -W -fPIC -D_REENTRANT -D_GNU_SOURCE -D_MAJOR_VERSION=2 -D_MINOR_VERSION=35 -D_BUILD_VERSION=0 -D_BUILD2_VERSION=2788 -pedantic -DMALLOC_TRACE -std=c++11 -I/usr/local/include -I/usr/local/include/opencv -L/usr/local/lib -L../../lib/arm64 -lopencv_highgui -lopencv_imgproc -lopencv_core -lstdc++ -lrt   -DNDEBUG -fvisibility=hidden    -I../../..   -ldl -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64:/usr/local/lib /usr/local/lib/libopencv_gapi.so.4.1.0 /usr/local/lib/libopencv_stitching.so.4.1.0 /usr/local/lib/libopencv_aruco.so.4.1.0 /usr/local/lib/libopencv_bgsegm.so.4.1.0 /usr/local/lib/libopencv_bioinspired.so.4.1.0 /usr/local/lib/libopencv_ccalib.so.4.1.0 /usr/local/lib/libopencv_cudabgsegm.so.4.1.0 /usr/local/lib/libopencv_cudafeatures2d.so.4.1.0 /usr/local/lib/libopencv_cudaobjdetect.so.4.1.0 /usr/local/lib/libopencv_cudastereo.so.4.1.0 /usr/local/lib/libopencv_dnn_objdetect.so.4.1.0 /usr/local/lib/libopencv_dpm.so.4.1.0 /usr/local/lib/libopencv_face.so.4.1.0 /usr/local/lib/libopencv_freetype.so.4.1.0 /usr/local/lib/libopencv_fuzzy.so.4.1.0 /usr/local/lib/libopencv_hfs.so.4.1.0 /usr/local/lib/libopencv_img_hash.so.4.1.0 /usr/local/lib/libopencv_line_descriptor.so.4.1.0 /usr/local/lib/libopencv_quality.so.4.1.0 /usr/local/lib/libopencv_reg.so.4.1.0 /usr/local/lib/libopencv_rgbd.so.4.1.0 /usr/local/lib/libopencv_saliency.so.4.1.0 /usr/local/lib/libopencv_stereo.so.4.1.0 /usr/local/lib/libopencv_structured_light.so.4.1.0 /usr/local/lib/libopencv_superres.so.4.1.0 /usr/local/lib/libopencv_surface_matching.so.4.1.0 /usr/local/lib/libopencv_tracking.so.4.1.0 /usr/local/lib/libopencv_videostab.so.4.1.0 /usr/local/lib/libopencv_xfeatures2d.so.4.1.0 /usr/local/lib/libopencv_xobjdetect.so.4.1.0 /usr/local/lib/libopencv_xphoto.so.4.1.0 /usr/local/lib/libopencv_shape.so.4.1.0 /usr/local/lib/libopencv_datasets.so.4.1.0 /usr/local/lib/libopencv_plot.so.4.1.0 /usr/local/lib/libopencv_text.so.4.1.0 /usr/local/lib/libopencv_dnn.so.4.1.0 /usr/local/lib/libopencv_ml.so.4.1.0 /usr/local/lib/libopencv_phase_unwrapping.so.4.1.0 /usr/local/lib/libopencv_cudacodec.so.4.1.0 /usr/local/lib/libopencv_cudaoptflow.so.4.1.0 /usr/local/lib/libopencv_cudalegacy.so.4.1.0 /usr/local/lib/libopencv_cudawarping.so.4.1.0 /usr/local/lib/libopencv_optflow.so.4.1.0 /usr/local/lib/libopencv_video.so.4.1.0 /usr/local/lib/libopencv_ximgproc.so.4.1.0 /usr/local/lib/libopencv_objdetect.so.4.1.0 /usr/local/lib/libopencv_calib3d.so.4.1.0 /usr/local/lib/libopencv_features2d.so.4.1.0 /usr/local/lib/libopencv_flann.so.4.1.0 /usr/local/lib/libopencv_highgui.so.4.1.0 /usr/local/lib/libopencv_videoio.so.4.1.0 /usr/local/lib/libopencv_imgcodecs.so.4.1.0 /usr/local/lib/libopencv_photo.so.4.1.0 /usr/local/lib/libopencv_cudaimgproc.so.4.1.0 /usr/local/lib/libopencv_cudafilters.so.4.1.0 /usr/local/lib/libopencv_imgproc.so.4.1.0 /usr/local/lib/libopencv_cudaarithm.so.4.1.0 /usr/local/lib/libopencv_core.so.4.1.0 /usr/local/lib/libopencv_cudev.so.4.1.0

echo "finished"
