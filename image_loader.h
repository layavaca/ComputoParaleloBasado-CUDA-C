#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <iostream>
#include <vector>
#include <string>

// Definimos esto para que stb_image implemente las funciones de lectura
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Función para cargar una imagen a color y convertirla a escala de grises (float)
inline bool loadGrayscaleImage(const std::string& filepath, std::vector<float>& h_out, int& width, int& height) {
    int channels;
    // Forzamos la carga a 3 canales (RGB) o 4 (RGBA) para asegurarnos de controlarlo
    unsigned char* img = stbi_load(filepath.c_str(), &width, &height, &channels, 3);
    
    if (img == nullptr) {
        std::cerr << "Error: No se pudo cargar la imagen desde " << filepath << std::endl;
        return false;
    }

    std::cout << "Imagen cargada con éxito. Dimensiones originales: " << width << "x" << height 
              << " con " << channels << " canales." << std::endl;

    h_out.resize(width * height);

    // Conversión a escala de grises usando los pesos de luminancia estándar (Rec. 601)
    for (int i = 0; i < width * height; ++i) {
        unsigned char r = img[i * 3 + 0];
        unsigned char g = img[i * 3 + 1];
        unsigned char b = img[i * 3 + 2];
        
        // Fórmula de luminancia: Y = 0.299*R + 0.587*G + 0.114*B
        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        h_out[i] = gray;
    }

    // Liberamos la memoria de la librería de imagen
    stbi_image_free(img);
    return true;
}

#endif