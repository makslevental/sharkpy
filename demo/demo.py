import sys
import ctypes
import importlib

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from mlir import ir
from mlir._mlir_libs import _mlir

import sharkpy

with sharkpy.ir.SharkBaseContext() as shark_ctx:
    registry = ir.DialectRegistry()
    m = importlib.import_module(
        f"mlir._mlir_libs._mlirRegisterEverything", "mlir._mlir_libs"
    )
    m.register_dialects(registry)
    shark_ctx.append_dialect_registry(registry)
    shark_ctx.load_all_available_dialects()
    print(shark_ctx.dialects["memref"])
    print(shark_ctx.get_dialect_namespace("memref"))
    print(shark_ctx.get_dialect_namespace("arith"))
