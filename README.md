# SharkPy

Shark dialect/compiler

## Install

pip doesn't work so you need to actually run `setup.py`, like so

```shell
$ python setup.py install
```

This will download a pre-built version of LLVM (and deposit them in `.sharkpy`), which includes object files (`.a`s) and
the python extensions.
If you get an error about 404 for the download then
check [makslevental/llvm-releases/](https://github.com/makslevental/llvm-releases/releases/) to see if there's a
pre-built package for your env.
If you already have these built somewhere then you can alternatively run

```shell
$ LLVM_INSTALL_DIR=<somewhere that has /include and /lib> python setup.py install
```

Afterwards you'll need to set `PYTHONPATH=$LLVM_INSTALL_DIR/python_packages/mlir_core`.

