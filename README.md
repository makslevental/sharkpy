# SharkPy

Shark dialect/compiler

## Install

pip doesn't work so you need to actually run `setup.py`, like so

```shell
$ python pybind_setup.py install
```

This will download a pre-built version of LLVM (and deposit them in `.sharkpy`), which includes object files (`.a`s) and
the python extensions.
If you already have these built somewhere then you can alternatively run

```shell
$ LLVM_INSTALL_DIR=<somewhere that has /include and /lib> python pybind_setup.py install
```

Afterwards you'll need to set `PYTHONPATH=$LLVM_INSTALL_DIR/python_packages/mlir_core` before running anything.

## Demo

`demo/demo.py` doesn't do much (yet):

```shell
$ LLVM_INSTALL_DIR=.sharkpy/llvm
$ PYTHONPATH=$LLVM_INSTALL_DIR/python_packages/mlir_core demo/demo.py

<Dialect memref (class mlir.dialects._memref_ops_gen._Dialect)>
is null 0
memref
112378063
is null 0
arith
112378354
 
```

but it does show off that you can successfully sublcass a python class and then access the underlying data.

## Gotchas

* If you get an error about 404 for the download then check [makslevental/llvm-releases/](https://github.com/makslevental/llvm-releases/releases/) to see if there's a
pre-built package for your env.

* If you get an error like

  ```shell
  Traceback (most recent call last):
    File "/home/mlevental/dev_projects/sharkpy/.sharkpy/llvm/python_packages/mlir_core/mlir/_mlir_libs/__init__.py", line 101, in <module>
      _site_initialize()
    File "/home/mlevental/dev_projects/sharkpy/.sharkpy/llvm/python_packages/mlir_core/mlir/_mlir_libs/__init__.py", line 56, in _site_initialize
      from ._mlir import ir
  ImportError: /home/mlevental/miniconda3/envs/sharkpy/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by   /home/mlevental/dev_projects/sharkpy/.sharkpy/llvm/python_packages/mlir_core/mlir/_mlir_libs/libMLIRPythonCAPI.so.15)
  ```
  
  then you need to upgrade `gcc` (or whatever compiler is your default compiler that's being used to compile the Pybind extension); you can also specify the compiler that setuptools uses by `CXX=clang++ CC=clang setup.py install`
* If you get an error 
  ```shell
  Traceback (most recent call last):
    File "/home/mlevental/dev_projects/sharkpy/demo/demo.py", line 10, in <module>
      import sharkpy
    File "/home/mlevental/miniconda3/envs/sharkpy/lib/python3.11/site-packages/sharkpy-0.0.1-py3.11-linux-x86_64.egg/sharkpy/__init__.py", line 2, in <module>
      from sharkpy_ext import *  # noqa: F403, F401
      ^^^^^^^^^^^^^^^^^^^^^^^^^
    ImportError: /home/mlevental/miniconda3/envs/sharkpy/lib/python3.11/site-packages/sharkpy-0.0.1-py3.11-linux-x86_64.egg/sharkpy_ext.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZN4llvm2cl18GenericOptionValue6anchorEv
  ```
  then you're in deep (some dll isn't being loaded or linked or doesn't have its symbols exported)

## Black magic

This works by combining [two pybind11-wrapped projects](https://github.com/pybind/pybind11/issues/2056) (this and MLIR's python bindings). 
Firstly, this requires `__attribute__ ((visibility ("default")))` on the symbols from MLIR's bindings ([to accomplish I cheat a little](https://github.com/makslevental/llvm-releases/blob/22e5924cc1b0a80576171d36477efab30ec218a7/build_llvm.bash#L117)).
Secondly, because of [C++ ABI nonsense](https://stackoverflow.com/a/67844737/9045206) this package and the MLIR bindings need to be compiled using the [same compiler and against the same version of STL](https://github.com/pybind/pybind11/issues/2056#issuecomment-570835972).
Since, I'm the one compiling the MLIR bindings I can control this but long term wheels will have to be built that lock those two together for both the MLIR bindings and this project.
More info [here](https://github.com/pybind/pybind11/issues/1193).