# Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280
Tiny-GPT on Vortex GPGPU for AMD Alveo U280 is a lightweight, end-to-end demo of GPT-style text generation running on the open-source RISC-V Vortex GPGPU architecture, deployed to an AMD/Xilinx Alveo U280 FPGA. The repo includes OpenCL device kernels and host code for a compact two-layer mat-vec inference loop with configurable temperature/top-k sampling, plus Python utilities to train and export small .npy weight files. It’s designed to showcase an open, reproducible AI/ML stack on FPGA—from building bitstreams and launching with XRT to capturing performance metrics (instructions, cycles, IPC) and exploring multicore scaling.

# Team Information

- Team number: AOHW25_616
- Project name: Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280
- University name: University of Essex
- Participant(s):
  -- Muhammadn Ahmed Khan
  -- Ya-lun Lee
  -- Qizal Arsalan
- Supervisor: Dr. Xiajun Zhai

# Vortex GPGPU

Vortex is a full-stack open-source RISC-V GPGPU. Vortex supports multiple backend drivers, including our C++ simulator (simx), an RTL simulator, and physical Xilinx and Altera FPGAs-- all controlled by a single driver script. The chosen driver determines the corresponding code invoked to run Vortex. Generally, developers will prototype their intended design in simx, before completing going forward with an RTL implementation. Alternatively, you can get up and running by selecting a driver of your choice and running a demo program.

## Credits & Upstream

This work builds on **Vortex GPGPU** (Apache-2.0): https://github.com/vortexgpgpu/vortex  
Major changes here: TinyGPT kernels, host flow, U280 configs, and build scripts.  
If you use this repo, please also cite the Vortex MICRO’21 paper (see upstream README).

## Specifications

- Support RISC-V RV32IMAF and RV64IMAFD
- Microarchitecture:
    - configurable number of cores, warps, and threads.
    - configurable number of ALU, FPU, LSU, and SFU units per core.
    - configurable pipeline issue width.
    - optional local memory, L1, L2, and L3 caches.
- Software:
    - OpenCL 1.2 Support.
- Supported FPGAs:
    - Altera Arria 10
    - Altera Stratix 10
    - Xilinx Alveo U50, U250, U280
    - Xilinx Versal VCK5000

## Directory structure

- `doc`: [Documentation](docs/index.md).
- `hw`: Hardware sources.
- `driver`: Host drivers repository.
- `runtime`: Kernel Runtime software.
- `sim`: Simulators repository.
- `tests`: Tests repository.
- `ci`: Continuous integration scripts.
- `miscs`: Miscellaneous resources.

## Build Instructions
More detailed build instructions can be found [here](docs/install_vortex.md).
### Supported OS Platforms
- Ubuntu 18.04, 20.04
- Centos 7
### Toolchain Dependencies
- [POCL](http://portablecl.org/)
- [LLVM](https://llvm.org/)
- [RISCV-GNU-TOOLCHAIN](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [Verilator](https://www.veripool.org/verilator)
- [FpNew](https://github.com/pulp-platform/fpnew.git)
- [SoftFloat](https://github.com/ucb-bar/berkeley-softfloat-3.git)
- [Ramulator](https://github.com/CMU-SAFARI/ramulator.git)
- [Yosys](https://github.com/YosysHQ/yosys)
- [Sv2v](https://github.com/zachjs/sv2v)
## Setup 
### Install Vortex codebase
```sh
git clone --depth=1 --recursive https://github.com/vortexgpgpu/vortex.git
cd vortex
```
### Install system dependencies
```sh
# ensure dependent libraries are present
sudo ./ci/install_dependencies.sh
```
### Configure your build folder
```sh
    mkdir build
    cd build
    # for 32bit
    ../configure --xlen=32 --tooldir=$HOME/tools
    # for 64bit
    ../configure --xlen=64 --tooldir=$HOME/tools
```
### Install prebuilt toolchain
```sh
./ci/toolchain_install.sh --all
```
### Set environment variables
```sh
# should always run before using the toolchain!
source ./ci/toolchain_env.sh
```
### Building Vortex
```sh
make -s
```
### Quick demo running vecadd OpenCL kernel on 2 cores
```sh
./ci/blackbox.sh --cores=2 --app=vecadd
```
### Build Tiny-gpt
```sh
# From Vortex root for V1
cd tests/opencl/tinygptv1
make clean
make
# From Vortex root for V1
cd tests/opencl/tinygptv1
make clean
make
```
### Run - Command for Sim
```sh
# For v2 from vortex root
./ci/blackbox.sh \
  --clusters=1 --cores=1 --warps=4 --threads=4 --driver=simx --app=tinygptv2 \
  --args="-engine persist  -tokens15\
  --perf=1

./ci/blackbox.sh \
  --clusters=1 --cores=4 --warps=4 --threads=4 --driver=simx --app=tinygptv2 \
  --args="-engine slice -groups 4 -tokens 15 -temp 0.8 -topk 5 -penalty 1.1" \
  --perf=1

./ci/blackbox.sh --clusters=1 --cores=1 --warps=4 --threads=4 --driver=simx --app=tinygptv2 \
  --args="-engine slice -groups 1 -tokens 15 -temp 0.8 -topk 5 -penalty 1.1" --perf=1

# For v1 from vortex root
./ci/blackbox.sh --clusters=1 --cores=1 --warps=4 --threads=4 --driver=simx --app=tinygptv1 \
--args="-temp 0.7 -topk 5 -penalty 1.1 -tokens 15"

./ci/blackbox.sh --clusters=1 --cores=2 --warps=4 --threads=4 --driver=simx --app=tinygptv1 \
--args="-temp 0.7 -topk 5 -penalty 1.1 -tokens 15"
```

### FPGA Execution
- To run tiny-gpt on FPGA hardware bitstream is required. For guidelines for bitstrean synthesis refer to docs/fpga_setup
- After generating bitsream from vrotex root run these commands for FPGA testing. Note: FPGA_BIN_DIR use location of your generated bitstrea
```sh
VX_FAST_EXIT=1 TARGET=hw FPGA_BIN_DIR=/tools/mk248883/vortex/hw/syn/xilinx/xrt/test1_xilinx_u280_gen3x16_xdma_1_202211_1_hw/bin ./ci/blackbox.sh --cores=1 --driver=xrt --app=tinygpt --args="-temp 0.7 -topk 5 -penalty 1.1 -tokens 15"

VX_FAST_EXIT=1 TARGET=hw FPGA_BIN_DIR=/tools/mk248883/vortex/hw/syn/xilinx/xrt/test2_xilinx_u280_gen3x16_xdma_1_202211_1_hw/bin ./ci/blackbox.sh --cores=2 --driver=xrt --app=tinygpt --args="-temp 0.7 -topk 5 -penalty 1.1 -tokens 15"
```
```
