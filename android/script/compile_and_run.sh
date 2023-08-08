########################
# compile Nerf project #
########################
./script/build-android.sh

#######################
# Run Demo on Android #
#######################
adb shell "mkdir -p /data/local/tmp/aotDemo"
adb shell "cd /data/local/tmp/aotDemo && rm -rf *"
adb push c_api/lib/libtaichi_c_api.so ../OpenCV-android-sdk/sdk/native/libs/arm64-v8a/libopencv_java4.so .tcm build-android-aarch64/aotDemo /data/local/tmp/aotDemo
adb shell "mkdir -p /data/local/tmp/aotDemo/build/assets/bench_case"
adb shell "cd /data/local/tmp/aotDemo/build/assets/bench_case && rm -rf *"
adb push ../taichi_algorithm/bench_case/*.tcm /data/local/tmp/aotDemo/build/assets/bench_case
adb push ../taichi_algorithm/bench_case/mountain.jpg /data/local/tmp/aotDemo/build/assets/bench_case
adb push ../taichi_algorithm/bench_case/house.jpg /data/local/tmp/aotDemo/build/assets/bench_case

adb shell "cd /data/local/tmp/aotDemo && LD_LIBRARY_PATH=. ./aotDemo"
