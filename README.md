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
### Install development tools
```sh
sudo apt-get install build-essential
sudo apt-get install binutils
sudo apt-get install python
sudo apt-get install uuid-dev
sudo apt-get install git
```
### Install Vortex codebase
```sh
git clone --depth=1 --recursive https://github.com/vortexgpgpu/vortex.git
cd vortex
```
### Configure your build folder
```sh
mkdir build
cd build
../configure --xlen=32 --tooldir=$HOME/tools
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

### Common Developer Tips
- Installing Vortex kernel and runtime libraries to use with external tools requires passing --prefix=<install-path> to the configure script.
```sh
../configure --xlen=32 --tooldir=$HOME/tools --prefix=<install-path>
make -s
make install
```
- Building Vortex 64-bit simply requires using --xlen=64 configure option.
```sh
../configure --xlen=32 --tooldir=$HOME/tools
```
- Sourcing "./ci/toolchain_env.sh" is required everytime you start a new terminal. we recommend adding "source <build-path>/ci/toolchain_env.sh" to your ~/.bashrc file to automate the process at login.
```sh
echo "source <build-path>/ci/toolchain_env.sh" >> ~/.bashrc
```
- Making changes to Makefiles in your source tree or adding new folders will require executing the "configure" script again to get it propagated into your build folder.
```sh
../configure
```
- To debug the GPU, you can generate a "run.log" trace. see /docs/debugging.md for more information.
```sh
./ci/blackbox.sh --app=demo --debug=3
```
- For additional information, check out the /docs.
