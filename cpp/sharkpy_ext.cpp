#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "IRModule.h"
#include "getenv.hpp"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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

template <typename cMlir_t, typename Mlir_t>
inline Mlir_t &unwrap(cMlir_t val) {
  assert(val.ptr && "unexpected non-null");
  return *(static_cast<Mlir_t *>(val.ptr));
}

template <typename Mlir_t, typename cMlir_t> inline cMlir_t wrap(Mlir_t &val) {
  return {&val};
}

#include <iostream>

void init_sharkpy_ext(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::object base_context =
      (py::object)py::module_::import("mlir._mlir_libs._mlir.ir")
          .attr("_BaseContext");

  mlir::python::adaptors::pure_subclass(m, "SharkBaseContext", base_context)
      .def(
          "get_dialect_namespace",
          [](mlir::python::PyMlirContext &self, std::string &name) {
            std::cerr << "is null " << mlirContextIsNull(self.get()) << "\n";
            auto dialect = unwrap<MlirContext, mlir::MLIRContext>(self.get())
                               .getLoadedDialect(name);
            std::cerr << dialect->getNamespace().str() << "\n";
            return hash_value(hash_value(dialect->getTypeID()));
          },
          py::arg("dialect_name"), "gets name space of dialect");
}

PYBIND11_MODULE(sharkpy_ext, m) {
  m.doc() = "Python bindings to Shark";
  m.def_submodule("ir");
  init_sharkpy_ext(m.def_submodule("ir"));
}