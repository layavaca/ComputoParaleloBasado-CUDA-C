#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Macro para verificar errores de CUDA de forma profesional
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// ==========================================
// KERNEL DE CUDA (Ejecutado en la GPU)
// ==========================================
__global__ void convolutionKernel(const float* d_in, float* d_out, int width, int height, 
                                  const float* d_mask, int maskSize) {
    // Transformación del índice global del hilo en coordenadas (columna, fila)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Control de límites: aseguramos que el hilo esté dentro de la imagen
    if (row < height && col < width) {
        float sum = 0.0f;
        int radius = maskSize / 2;

        // Recorrido de la vecindad de la máscara
        for (int mi = -radius; mi <= radius; ++mi) {
            for (int mj = -radius; mj <= radius; ++mj) {
                int currRow = row + mi;
                int currCol = col + mj;

                // Condición común: Aplicación de extensión con ceros (zero-padding) fuera de la imagen
                if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width) {
                    float pixel = d_in[currRow * width + currCol];
                    float weight = d_mask[(mi + radius) * maskSize + (mj + radius)];
                    sum += pixel * weight;
                }
            }
        }
        // Guardar el resultado en el arreglo de salida en memoria global
        d_out[row * width + col] = sum;
    }
}

// ==========================================
// FUNCIÓNN HOST PARA LANZAR EL KERNEL Y MEDIR TIEMPOS
// ==========================================
void runCudaConvolution(const std::vector<float>& h_in, std::vector<float>& h_out, 
                        int width, int height, const std::vector<float>& h_mask, int maskSize,
                        float& timeH2D, float& timeKernel, float& timeD2H) {
    
    size_t imgBytes = width * height * sizeof(float);
    size_t maskBytes = maskSize * maskSize * sizeof(float);

    float *d_in, *d_out, *d_mask;

    // 1. Reserva de memoria en la GPU
    CHECK_CUDA(cudaMalloc(&d_in, imgBytes));
    CHECK_CUDA(cudaMalloc(&d_out, imgBytes));
    CHECK_CUDA(cudaMalloc(&d_mask, maskBytes));

    // Eventos de CUDA para medición precisa de tiempo
    cudaEvent_t startEvt, stopEvt;
    cudaEventCreate(&startEvt);
    cudaEventCreate(&stopEvt);

    // 2. Transferencia Host -> Device (CPU a GPU)
    cudaEventRecord(startEvt);
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), imgBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mask, h_mask.data(), maskBytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stopEvt);
    cudaEventSynchronize(stopEvt);
    cudaEventElapsedTime(&timeH2D, startEvt, stopEvt);

    // 3. Configuración de Bloques e Hilos en X e Y
    // Usaremos bloques de 16x16 (256 hilos por bloque, ideal para empezar)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 4. Ejecución del Kernel en la GPU
    cudaEventRecord(startEvt);
    convolutionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, width, height, d_mask, maskSize);
    // Asegurar finalización del trabajo en la GPU
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(stopEvt);
    cudaEventSynchronize(stopEvt);
    cudaEventElapsedTime(&timeKernel, startEvt, stopEvt);

    // 5. Transferencia Device -> Host (GPU a CPU)
    cudaEventRecord(startEvt);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, imgBytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stopEvt);
    cudaEventSynchronize(stopEvt);
    cudaEventElapsedTime(&timeD2H, startEvt, stopEvt);

    // Liberación de recursos de la GPU
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_mask);
    cudaEventDestroy(startEvt);
    cudaEventDestroy(stopEvt);
}