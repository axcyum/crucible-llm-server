"""Reuse the pinned upstream framework helpers, building only the iOS device slice."""
from pathlib import Path
import subprocess

root = Path("llama.cpp")
source = (root / "build-xcframework.sh").read_text()
helpers = source.split('echo "Building for iOS simulator..."', 1)[0]
device = source.split('echo "Building for iOS devices..."', 1)[1].split('echo "Building for macOS..."', 1)[0]
assert "cmake -B build-ios-device" in device
script = helpers + '\necho "Building for iOS devices..."\n' + device + r'''
setup_framework_structure "build-ios-device" ${IOS_MIN_OS_VERSION} "ios"
combine_static_libraries "build-ios-device" "Release-iphoneos" "ios" "false"
xcrun xcodebuild -create-xcframework \
  -framework "$(pwd)/build-ios-device/framework/llama.framework" \
  -debug-symbols "$(pwd)/build-ios-device/dSYMs/llama.dSYM" \
  -output "$(pwd)/build-apple/llama.xcframework"
'''
(root / "build-ios-only.sh").write_text(script)
subprocess.run(["bash", "build-ios-only.sh"], cwd=root, check=True)
