#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>

// Incluimos nuestros módulos anteriores
#include "image_loader.h"
#include "image_utils.h"

// Declaración de la función externa escrita en CUDA (.cu)
void runCudaConvolution(const std::vector<float>& h_in, std::vector<float>& h_out, 
                        int width, int height, const std::vector<float>& h_mask, int maskSize,
                        float& timeH2D, float& timeKernel, float& timeD2H);

int main(int argc, char* argv[]) {
    // Verificamos si se pasó la ruta de la imagen por argumento
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " <ruta_imagen.jpg/png>\n";
        std::cout << "Ejecutando con imagen de prueba simulada si no se provee archivo...\n";
    }

    std::string imagePath = (argc > 1) ? argv[1] : "";
    std::vector<float> h_in;
    int width = 0, height = 0;

    // Intentamos cargar la imagen real; si falla o no se pasa, creamos una matriz sintética para probar
    if (imagePath.empty() || !loadGrayscaleImage(imagePath, h_in, width, height)) {
        width = 2000;
        height = 2000;
        h_in.resize(width * height, 150.0f); // Imagen sintética de prueba
        std::cout << "Usando matriz sintética de prueba de " << width << "x" << height << "\n";
    }

    std::vector<float> h_out_cpu(width * height, 0.0f);
    std::vector<float> h_out_gpu(width * height, 0.0f);

    // ==========================================
    // DEFINICIÓN DE LOS 3 FILTROS EXIGIDOS (3x3)
    // ==========================================
    int maskSize = 3;

    // 1. Filtro de Media (Suavizado)
    std::vector<float> maskMedia = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };

    // 2. Filtro de Realce de Nitidez (Sharpening)
    std::vector<float> maskNitidez = {
         0.0f, -1.0f,  0.0f,
        -1.0f,  5.0f, -1.0f,
         0.0f, -1.0f,  0.0f
    };

    // 3. Filtro Laplaciano (Detección de cambios de intensidad)
    std::vector<float> maskLaplaciano = {
        0.0f,  1.0f,  0.0f,
        1.0f, -4.0f,  1.0f,
        0.0f,  1.0f,  0.0f
    };

    // Seleccionamos el filtro a ejecutar (ej. Media)
    std::vector<float> activeMask = maskMedia;
    std::string filterName = "Media (3x3)";

    std::cout << "\n--- Ejecutando filtro: " << filterName << " ---" << std::endl;

    // ==========================================
    // EJECUCIÓN SECUENCIAL (CPU) - Referencia
    // ==========================================
    auto startCpu = std::chrono::high_resolution_clock::now();
    convolutionCPU(h_in, h_out_cpu, width, height, activeMask, maskSize);
    auto endCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = endCpu - startCpu;
    std::cout << "Tiempo CPU (Secuencial): " << cpuTime.count() << " ms\n";

    // ==========================================
    // EJECUCIÓN PARALELA (GPU - CUDA)
    // ==========================================
    float timeH2D = 0.0f, timeKernel = 0.0f, timeD2H = 0.0f;
    runCudaConvolution(h_in, h_out_gpu, width, height, activeMask, maskSize, timeH2D, timeKernel, timeD2H);

    std::cout << "Tiempo Transferencia H2D: " << timeH2D << " ms\n";
    std::cout << "Tiempo Kernel CUDA:       " << timeKernel << " ms\n";
    std::cout << "Tiempo Transferencia D2H: " << timeD2H << " ms\n";
    std::cout << "Tiempo Conjunto (GPU):    " << (timeH2D + timeKernel + timeD2H) << " ms\n";

    // Aceleraciones solicitadas en las instrucciones
    double speedupKernel = cpuTime.count() / timeKernel;
    double speedupTotal = cpuTime.count() / (timeH2D + timeKernel + timeD2H);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Aceleración de cómputo (CPU / Kernel): " << speedupKernel << "x\n";
    std::cout << "Aceleración con transferencias:       " << speedupTotal << "x\n";

    // ==========================================
    // VALIDACIÓN NUMÉRICA RIGUROSA CPU vs GPU
    // ==========================================
    float maxError = 0.0f;
    float sumError = 0.0f;
    size_t totalPixels = width * height;

    for (size_t i = 0; i < totalPixels; ++i) {
        float err = std::abs(h_out_cpu[i] - h_out_gpu[i]);
        if (err > maxError) {
            maxError = err;
        }
        sumError += err;
    }
    float meanError = sumError / totalPixels;

    std::cout << "\n--- Validación Numérica ---" << std::endl;
    std::cout << "Error absoluto máximo: " << maxError << "\n";
    std::cout << "Error absoluto medio:  " << meanError << "\n";

    // Definimos una tolerancia estricta para números flotantes en operaciones de convolución
    float tolerancia = 1e-4f;
    if (maxError < tolerancia) {
        std::cout << "¡VALIDACIÓN EXITOSA! Los resultados de CPU y GPU coinciden dentro de la tolerancia (" << tolerancia << ").\n";
    } else {
        std::cout << "¡ADVERTENCIA! El error supera la tolerancia esperada. Revisar precisión numérica.\n";
    }

    return 0;
}