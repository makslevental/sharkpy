import os
import platform
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import NamedTuple

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


# Taken from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_build_type():
    if check_env_flag("DEBUG") or check_env_flag("REL_WITH_DEB_INFO"):
        return "Debug"
    else:
        return "Release"


def get_llvm():
    match platform.system():
        case "Linux":
            system_suffix = "linux-gnu-ubuntu-18.04"
        case "Darwin":
            system_suffix = "apple-darwin"
        case other:
            raise NotImplementedError(f"unknown system {other}")

    match get_build_type():
        case "Debug":
            release_or_debug = "assert"
        case "Release":
            release_or_debug = "release"

    name = f"llvm+mlir-15.0.0-{platform.machine()}-{system_suffix}-{release_or_debug}"
    name = f"llvm+mlir-15.0.0-x86_64-{system_suffix}-{release_or_debug}"
    url = f"https://github.com/shintaro-iwasaki/llvm-releases/releases/download/llvm-15.0.0-4ba6a9c9f65b/{name}.tar.xz"
    url = f"https://github.com/shintaro-iwasaki/llvm-releases/releases/download/llvm-15.0.0-4ba6a9c9f65b/llvm+mlir-15.0.0-x86_64-apple-darwin-release.tar.xz"
    package_root_dir = Path(".sharkpy") / "llvm"
    package_dir = package_root_dir / name
    test_file_path = package_dir / "lib"
    if not test_file_path.exists():
        try:
            shutil.rmtree(package_root_dir)
        except Exception:
            pass
        package_root_dir.mkdir(parents=True, exist_ok=True)
        print("downloading and extracting {} ...".format(url))
        ftpstream = urllib.request.urlopen(url)
        file = tarfile.open(fileobj=ftpstream, mode="r|*")
        file.extractall(path=str(package_root_dir))

    return Path(package_dir).absolute()


llvm_package_dir: Path = get_llvm()


setup(
    name="sharkpy",
    version="0.0.1",
    long_description="",
    ext_modules=[
        Pybind11Extension(
            "sharkpy_ext",
            [
                "cpp/sharkpy.cpp"
            ],  # Example: passing in the version to the compiled code
            # define_macros=[("VERSION_INFO", __version__)],
            include_dirs=[
                str(llvm_package_dir / "include"),
            ],
            libraries=[],
            library_dirs=[str(llvm_package_dir / "lib")],  # extra_link_args=libs
        ),
    ],
    packages=["sharkpy"],
    package_dir={"sharkpy": "python"},  # package_data={
    #     'pedalboard': ['py.typed', '*.pyi', '**/*.pyi']
    # },
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
