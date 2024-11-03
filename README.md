# Tiled-Matrix-Multiplication

## Author
- Name: [Your Full Name]
- UCInetID: [Your UCInetID]

## Overview
This project implements a CUDA program for tiled matrix multiplication, designed to handle input matrices of arbitrary sizes. The implementation includes CUDA kernels, a host program, and configurable runtime parameters such as grid sizes and thread block sizes. The program tests the correctness of GPU results against a CPU-based matrix multiplication and measures execution times for both implementations.

## Project Structure
The repository contains the following files:
- **hpatloll_lab1.cu**: The main source file containing the CUDA kernel code, the `MatrixMultiplication` function, and host code.
- **Makefile**: A makefile to compile the CUDA source code.
- **readme.md**: This documentation file with project details and instructions.

## Implementation Details
### Key Components
1. **CUDA Kernel**: Implements tiled matrix multiplication using shared memory to optimize memory access and improve parallelism.
2. **Top Function (`MatrixMultiplication`)**: Configures runtime parameters, launches the CUDA kernel, and checks the results.
3. **Host Code**: Initializes matrices, allocates memory on the device, and compares GPU results with CPU results.

### Runtime Parameters
- **Grid Size**: Configured based on the input matrix size.
- **Block Size**: Defined as per the device capabilities (e.g., 16x16 or 32x32 tiles).

### Memory Hierarchy
The memory allocation can be adjusted, but the evaluation code on the host side must remain unaffected to ensure consistent testing.

## Compilation Instructions
### Prerequisites
- Ensure that CUDA is properly installed on your system.
- Verify that `nvcc` (the CUDA compiler) is available.

### Build Steps
To compile the program, use the following command:
```bash
make
```

### Running the Program
Run the compiled executable as follows:
```bash
./hpatloll_lab1
```

### Clean Up
To clean up generated files, use:
```bash
make clean
```

## Testing and Verification
The program includes tests with randomly generated matrices to validate the GPU implementation against a CPU-based reference. The results are compared using a floating-point tolerance threshold to account for precision differences.

## Performance Measurement
Execution times for both CPU and GPU implementations are printed in the output to facilitate performance analysis.
