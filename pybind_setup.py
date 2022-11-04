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
            system_suffix = "linux-gnu-ubuntu-20.04"
        case "Darwin":
            system_suffix = "apple-darwin"
        case other:
            raise NotImplementedError(f"unknown system {other}")

    match get_build_type():
        case "Debug":
            release_or_debug = "assert"
        case "Release":
            release_or_debug = "release"

    major, minor, _ = platform.python_version_tuple()
    name = f"llvm+mlir+python-{major}.{minor}-15.0.0-{platform.machine()}-{system_suffix}-{release_or_debug}"
    url = f"https://github.com/makslevental/llvm-releases/releases/download/llvm-15.0.0-4ba6a9c9f65b/{name}.tar.xz"
    package_root_dir = Path(".sharkpy")
    if package_dir := os.getenv("LLVM_INSTALL_DIR"):
        package_dir = Path(package_dir)
    else:
        package_dir = package_root_dir / name
    test_file_path = package_dir / "lib"
    if not test_file_path.exists():
        try:
            shutil.rmtree(package_root_dir)
        except Exception:
            pass
        package_root_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading and extracting {url} ...")
        ftpstream = urllib.request.urlopen(url)
        file = tarfile.open(fileobj=ftpstream, mode="r|*")
        file.extractall(path=str(package_root_dir))

    return Path(package_dir).absolute()


llvm_package_dir: Path = get_llvm()

# https://stackoverflow.com/a/20728782
libs = [
    "LLVMAArch64AsmParser",
    "LLVMAArch64CodeGen",
    "LLVMAArch64Desc",
    "LLVMAArch64Disassembler",
    "LLVMAArch64Info",
    "LLVMAArch64Utils",
    "LLVMAggressiveInstCombine",
    "LLVMAnalysis",
    "LLVMAsmParser",
    "LLVMAsmPrinter",
    "LLVMBinaryFormat",
    "LLVMBitReader",
    "LLVMBitWriter",
    "LLVMBitstreamReader",
    "LLVMCFGuard",
    "LLVMCFIVerify",
    "LLVMCodeGen",
    "LLVMCore",
    "LLVMCoroutines",
    "LLVMCoverage",
    "LLVMDWARFLinker",
    "LLVMDWP",
    "LLVMDebugInfoCodeView",
    "LLVMDebugInfoDWARF",
    "LLVMDebugInfoGSYM",
    "LLVMDebugInfoMSF",
    "LLVMDebugInfoPDB",
    "LLVMDebuginfod",
    "LLVMDemangle",
    "LLVMDiff",
    "LLVMDlltoolDriver",
    "LLVMExecutionEngine",
    "LLVMExegesis",
    "LLVMExegesisAArch64",
    "LLVMExtensions",
    "LLVMFileCheck",
    "LLVMFrontendOpenACC",
    "LLVMFrontendOpenMP",
    "LLVMFuzzMutate",
    "LLVMFuzzerCLI",
    "LLVMGlobalISel",
    "LLVMIRReader",
    "LLVMInstCombine",
    "LLVMInstrumentation",
    "LLVMInterfaceStub",
    "LLVMInterpreter",
    "LLVMJITLink",
    "LLVMLTO",
    "LLVMLibDriver",
    "LLVMLineEditor",
    "LLVMLinker",
    "LLVMMC",
    "LLVMMCA",
    "LLVMMCDisassembler",
    "LLVMMCJIT",
    "LLVMMCParser",
    "LLVMMIRParser",
    "LLVMObjCARCOpts",
    "LLVMObjCopy",
    "LLVMObject",
    "LLVMObjectYAML",
    "LLVMOption",
    "LLVMOrcJIT",
    "LLVMOrcShared",
    "LLVMOrcTargetProcess",
    "LLVMPasses",
    "LLVMProfileData",
    "LLVMRemarks",
    "LLVMRuntimeDyld",
    "LLVMScalarOpts",
    "LLVMSelectionDAG",
    "LLVMSupport",
    "LLVMSymbolize",
    "LLVMTableGen",
    "LLVMTableGenGlobalISel",
    "LLVMTarget",  # "LLVMTestingSupport",
    "LLVMTextAPI",
    "LLVMTransformUtils",
    "LLVMVectorize",
    "LLVMWindowsDriver",
    "LLVMWindowsManifest",
    "LLVMXRay",
    "LLVMipo",
    "MLIRAMDGPUDialect",
    "MLIRAMDGPUToROCDL",
    "MLIRAMXDialect",
    "MLIRAMXToLLVMIRTranslation",
    "MLIRAMXTransforms",
    "MLIRAffineAnalysis",
    "MLIRAffineDialect",
    "MLIRAffineToStandard",
    "MLIRAffineTransforms",
    # "MLIRAffineTransformsTestPasses",
    "MLIRAffineUtils",
    "MLIRAnalysis",
    "MLIRArithmeticDialect",
    "MLIRArithmeticToLLVM",
    "MLIRArithmeticToSPIRV",
    "MLIRArithmeticTransforms",
    "MLIRArithmeticUtils",
    "MLIRArmNeon2dToIntr",
    "MLIRArmNeonDialect",
    "MLIRArmNeonToLLVMIRTranslation",
    "MLIRArmSVEDialect",
    "MLIRArmSVEToLLVMIRTranslation",
    "MLIRArmSVETransforms",
    "MLIRAsmParser",
    "MLIRAsyncDialect",
    "MLIRAsyncToLLVM",
    "MLIRAsyncTransforms",
    "MLIRBufferizationDialect",
    "MLIRBufferizationToMemRef",
    "MLIRBufferizationTransformOps",
    "MLIRBufferizationTransforms",
    "MLIRCAPIAsync",
    "MLIRCAPIControlFlow",
    "MLIRCAPIConversion",
    "MLIRCAPIDebug",
    "MLIRCAPIExecutionEngine",
    "MLIRCAPIFunc",
    "MLIRCAPIGPU",
    "MLIRCAPIIR",
    "MLIRCAPIInterfaces",
    "MLIRCAPILLVM",
    "MLIRCAPILinalg",
    "MLIRCAPIPDL",
    # "MLIRCAPIPythonTestDialect",
    "MLIRCAPIQuant",
    "MLIRCAPIRegisterEverything",
    "MLIRCAPISCF",
    "MLIRCAPIShape",
    "MLIRCAPISparseTensor",
    "MLIRCAPITensor",
    "MLIRCAPITransforms",
    "MLIRCallInterfaces",
    "MLIRCastInterfaces",
    "MLIRComplexDialect",
    "MLIRComplexToLLVM",
    "MLIRComplexToLibm",
    "MLIRComplexToStandard",
    "MLIRControlFlowDialect",
    "MLIRControlFlowInterfaces",
    "MLIRControlFlowToLLVM",
    "MLIRControlFlowToSPIRV",
    "MLIRCopyOpInterface",
    "MLIRDLTIDialect",  # "MLIRDLTITestPasses",
    "MLIRDataLayoutInterfaces",
    "MLIRDerivedAttributeOpInterface",
    "MLIRDialect",
    "MLIRDialectUtils",
    "MLIREmitCDialect",
    "MLIRExecutionEngine",
    "MLIRExecutionEngineUtils",
    "MLIRFuncDialect",  # "MLIRFuncTestPasses",
    "MLIRFuncToLLVM",
    "MLIRFuncToSPIRV",
    "MLIRFuncTransforms",
    "MLIRGPUOps",  # "MLIRGPUTestPasses",
    "MLIRGPUToGPURuntimeTransforms",
    "MLIRGPUToNVVMTransforms",
    "MLIRGPUToROCDLTransforms",
    "MLIRGPUToSPIRV",
    "MLIRGPUToVulkanTransforms",
    "MLIRGPUTransforms",
    "MLIRIR",
    "MLIRInferIntRangeInterface",
    "MLIRInferTypeOpInterface",
    "MLIRJitRunner",
    "MLIRLLVMCommonConversion",
    "MLIRLLVMDialect",
    "MLIRLLVMIRTransforms",
    "MLIRLLVMToLLVMIRTranslation",
    "MLIRLinalgAnalysis",
    "MLIRLinalgDialect",  # "MLIRLinalgTestPasses",
    "MLIRLinalgToLLVM",
    "MLIRLinalgToSPIRV",
    "MLIRLinalgToStandard",
    "MLIRLinalgTransformOps",
    "MLIRLinalgTransforms",
    "MLIRLinalgUtils",
    "MLIRLoopLikeInterface",
    "MLIRLspServerLib",
    "MLIRLspServerSupportLib",
    "MLIRMLProgramDialect",
    "MLIRMathDialect",  # "MLIRMathTestPasses",
    "MLIRMathToLLVM",
    "MLIRMathToLibm",
    "MLIRMathToSPIRV",
    "MLIRMathTransforms",
    "MLIRMemRefDialect",
    # "MLIRMemRefTestPasses",
    "MLIRMemRefToLLVM",
    "MLIRMemRefToSPIRV",
    "MLIRMemRefTransforms",
    "MLIRMemRefUtils",
    "MLIRMlirOptMain",
    "MLIRNVGPUDialect",
    "MLIRNVGPUToNVVM",
    "MLIRNVGPUTransforms",
    "MLIRNVVMDialect",
    "MLIRNVVMToLLVMIRTranslation",
    "MLIROpenACCDialect",
    "MLIROpenACCToLLVM",
    "MLIROpenACCToLLVMIRTranslation",
    "MLIROpenACCToSCF",
    "MLIROpenMPDialect",
    "MLIROpenMPToLLVM",
    "MLIROpenMPToLLVMIRTranslation",
    "MLIROptLib",  # "MLIRPDLDialect",
    # "MLIRPDLInterpDialect",
    # "MLIRPDLLAST",
    # "MLIRPDLLCodeGen",
    # "MLIRPDLLODS",
    # "MLIRPDLLParser",
    # "MLIRPDLToPDLInterp",
    "MLIRParallelCombiningOpInterface",
    "MLIRParser",
    "MLIRPass",  # "MLIRPdllLspServerLib",
    "MLIRPresburger",  # "MLIRPythonTestDialect",
    "MLIRQuantDialect",
    "MLIRQuantTransforms",
    "MLIRQuantUtils",
    "MLIRROCDLDialect",
    "MLIRROCDLToLLVMIRTranslation",
    "MLIRReconcileUnrealizedCasts",
    "MLIRReduce",
    "MLIRReduceLib",
    "MLIRRewrite",
    "MLIRSCFDialect",
    # "MLIRSCFTestPasses",
    "MLIRSCFToControlFlow",
    "MLIRSCFToGPU",
    "MLIRSCFToOpenMP",
    "MLIRSCFToSPIRV",
    "MLIRSCFTransformOps",
    "MLIRSCFTransforms",
    "MLIRSCFUtils",
    "MLIRSPIRVBinaryUtils",
    "MLIRSPIRVConversion",
    "MLIRSPIRVDeserialization",
    "MLIRSPIRVDialect",
    "MLIRSPIRVModuleCombiner",
    "MLIRSPIRVSerialization",  # "MLIRSPIRVTestPasses",
    "MLIRSPIRVToLLVM",
    "MLIRSPIRVTransforms",
    "MLIRSPIRVTranslateRegistration",
    "MLIRSPIRVUtils",
    "MLIRShapeDialect",
    "MLIRShapeOpsTransforms",  # "MLIRShapeTestPasses",
    "MLIRShapeToStandard",
    "MLIRSideEffectInterfaces",
    "MLIRSparseTensorDialect",
    "MLIRSparseTensorPipelines",
    "MLIRSparseTensorTransforms",
    "MLIRSparseTensorUtils",
    "MLIRSupport",
    "MLIRSupportIndentedOstream",
    "MLIRTableGen",
    "MLIRTargetCpp",
    "MLIRTargetLLVMIRExport",
    "MLIRTargetLLVMIRImport",
    "MLIRTensorDialect",
    "MLIRTensorInferTypeOpInterfaceImpl",  # "MLIRTensorTestPasses",
    "MLIRTensorTilingInterfaceImpl",
    "MLIRTensorToLinalg",
    "MLIRTensorToSPIRV",
    "MLIRTensorTransforms",
    "MLIRTensorUtils",  # "MLIRTestAnalysis",
    # "MLIRTestDialect",
    # "MLIRTestFuncToLLVM",
    # "MLIRTestIR",
    # "MLIRTestPDLL",
    # "MLIRTestPass",
    # "MLIRTestReducer",
    # "MLIRTestRewrite",
    # "MLIRTestTransformDialect",
    # "MLIRTestTransforms",
    "MLIRTilingInterface",  # "MLIRTilingInterfaceTestPasses",
    "MLIRToLLVMIRTranslationRegistration",
    "MLIRTosaDialect",  # "MLIRTosaTestPasses",
    "MLIRTosaToArith",
    "MLIRTosaToLinalg",
    "MLIRTosaToSCF",
    "MLIRTosaToTensor",
    "MLIRTosaTransforms",
    "MLIRTransformDialect",
    "MLIRTransformDialectTransforms",
    "MLIRTransformUtils",
    "MLIRTransforms",
    "MLIRTranslateLib",
    "MLIRVectorDialect",
    "MLIRVectorInterfaces",  # "MLIRVectorTestPasses",
    "MLIRVectorToGPU",
    "MLIRVectorToLLVM",
    "MLIRVectorToSCF",
    "MLIRVectorToSPIRV",
    "MLIRVectorTransforms",
    "MLIRVectorUtils",
    "MLIRViewLikeInterface",
    "MLIRX86VectorDialect",
    "MLIRX86VectorToLLVMIRTranslation",
    "MLIRX86VectorTransforms",
]

setup(
    name="sharkpy",
    version="0.0.1",
    long_description="",
    ext_modules=[
        Pybind11Extension(
            "sharkpy_ext",
            [
                "cpp/sharkpy_ext.cpp"
            ],  # Example: passing in the version to the compiled code
            # define_macros=[("VERSION_INFO", __version__)],
            include_dirs=[
                str(llvm_package_dir / "include"),
            ],
            libraries=libs,
            library_dirs=[str(llvm_package_dir / "lib")],  # extra_link_args=libs
        ),
    ],
    packages=["sharkpy"],
    package_dir={"sharkpy": "python"},
    # package_data={
    #     'pedalboard': ['py.typed', '*.pyi', '**/*.pyi']
    # },
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
