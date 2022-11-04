#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "IRModule.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/SourceMgr.h"

#include <Python.h>
#include <cctype>
#include <fstream>
#include <optional>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

template<typename cMlirType, typename mlirType>
inline mlirType &unwrap(cMlirType val) {
    assert(val.ptr && "unexpected non-null");
    return *(static_cast<mlirType *>(val.ptr));
}

template<typename cMlirType, typename mlirType>
inline cMlirType wrap(mlirType &val) {
    return {&val};
}

struct Dog {
};

void init_sharkpy_ext(py::module &&m) {
    using ret = py::return_value_policy;
    using namespace pybind11::literals;

    py::object _context = (py::object) py::module_::import("mlir._mlir_libs._mlir.ir").attr("Context");

    py::class_<Dog>(m, "Dog", _context).def(py::init<>()).def("bark", [](const Dog &self) { return "bark"; });
}


PYBIND11_MODULE(sharkpy_ext, m) {
    m.doc() = "Python bindings to Shark";
    m.def_submodule("ir");
    init_sharkpy_ext(m.def_submodule("ir"));
}