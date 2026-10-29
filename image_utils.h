#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <vector>
#include <cmath>
#include <iostream>

// Versión secuencial en CPU para usar como referencia
inline void convolutionCPU(const std::vector<float>& h_in, std::vector<float>& h_out, 
                           int width, int height, const std::vector<float>& h_mask, int maskSize) {
    int radius = maskSize / 2;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            float sum = 0.0f;
            for (int mi = -radius; mi <= radius; ++mi) {
                for (int mj = -radius; mj <= radius; ++mj) {
                    int currRow = row + mi;
                    int currCol = col + mj;
                    
                    // Extensión con ceros fuera de los límites de la imagen
                    if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width) {
                        float pixel = h_in[currRow * width + currCol];
                        float weight = h_mask[(mi + radius) * maskSize + (mj + radius)];
                        sum += pixel * weight;
                    }
                }
            }
            h_out[row * width + col] = sum;
        }
    }
}

#endif