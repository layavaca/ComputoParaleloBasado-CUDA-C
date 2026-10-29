#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <string>

using namespace std;

// ============================================================
// ESTRUCTURA PARA REPRESENTAR UNA MASCARA DE CONVOLUCION
// ============================================================

struct Mascara
{
    string nombre;
    int tamano;
    vector<float> coeficientes;
};

// ============================================================
// FILTRO DE CONVOLUCION SECUENCIAL
//
// Cada iteracion calcula un pixel de salida.
// Los pixeles fuera de la imagen se consideran cero.
// La entrada y la salida utilizan arreglos separados.
// ============================================================

void convolucionCPU(
    const vector<float> &entrada,
    vector<float> &salida,
    int ancho,
    int alto,
    const Mascara &mascara)
{
    if (ancho <= 0 || alto <= 0)
    {
        throw invalid_argument("Dimensiones invalidas.");
    }

    if (entrada.size() != static_cast<size_t>(ancho) * alto)
    {
        throw invalid_argument("El tamano de entrada no coincide.");
    }

    if (mascara.tamano <= 0 ||
        mascara.tamano % 2 == 0 ||
        mascara.coeficientes.size() !=
            static_cast<size_t>(mascara.tamano) * mascara.tamano)
    {
        throw invalid_argument("Mascara invalida.");
    }

    salida.assign(entrada.size(), 0.0f);

    int radio = mascara.tamano / 2;

    // Recorrer las filas de la imagen.
    for (int y = 0; y < alto; y++)
    {

        // Recorrer las columnas de la imagen.
        for (int x = 0; x < ancho; x++)
        {

            float acumulador = 0.0f;

            // Recorrer las filas de la mascara.
            for (int ky = -radio; ky <= radio; ky++)
            {

                // Recorrer las columnas de la mascara.
                for (int kx = -radio; kx <= radio; kx++)
                {

                    int fila = y + ky;
                    int columna = x + kx;

                    // Extension con ceros:
                    // ignorar las posiciones fuera de la imagen.
                    if (fila < 0 || fila >= alto ||
                        columna < 0 || columna >= ancho)
                    {
                        continue;
                    }

                    int indiceImagen = fila * ancho + columna;

                    int indiceMascara =
                        (ky + radio) * mascara.tamano + (kx + radio);

                    float pixel = entrada[indiceImagen];

                    float coeficiente =
                        mascara.coeficientes[indiceMascara];

                    acumulador += pixel * coeficiente;
                }
            }

            int indiceSalida = y * ancho + x;

            salida[indiceSalida] = acumulador;
        }
    }
}

// ============================================================
// IMPRIMIR UNA IMAGEN PEQUENA
// ============================================================

void imprimirImagen(
    const vector<float> &imagen,
    int ancho,
    int alto)
{
    for (int y = 0; y < alto; y++)
    {

        for (int x = 0; x < ancho; x++)
        {

            cout << setw(10)
                 << fixed << setprecision(2)
                 << imagen[y * ancho + x];
        }

        cout << '\n';
    }
}

// ============================================================
// PROGRAMA PRINCIPAL
// ============================================================

int main()
{

    // Imagen de prueba de 5 x 5 pixeles.
    const int ancho = 5;
    const int alto = 5;

    vector<float> imagen = {
        10, 20, 30, 40, 50,
        20, 40, 60, 80, 100,
        30, 60, 90, 120, 150,
        40, 80, 120, 160, 200,
        50, 100, 150, 200, 250};

    // Mascara de media 3 x 3.
    Mascara media = {
        "Media 3x3",
        3,
        {1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
         1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
         1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f}};

    // Mascara de realce de nitidez 3 x 3.
    Mascara nitidez = {
        "Nitidez 3x3",
        3,
        {0, -1, 0,
         -1, 5, -1,
         0, -1, 0}};

    // Mascara laplaciana 3 x 3.
    Mascara laplaciano = {
        "Laplaciano 3x3",
        3,
        {0, 1, 0,
         1, -4, 1,
         0, 1, 0}};

    vector<float> salida;

    cout << "====================================\n";
    cout << " CONVOLUCION SECUENCIAL EN CPU\n";
    cout << "====================================\n";

    cout << "\nImagen original:\n";
    imprimirImagen(imagen, ancho, alto);

    // Ejecutar el filtro de media.
    convolucionCPU(imagen, salida, ancho, alto, media);

    cout << "\nResultado: " << media.nombre << "\n";
    imprimirImagen(salida, ancho, alto);

    // Ejecutar el filtro de nitidez.
    convolucionCPU(imagen, salida, ancho, alto, nitidez);

    cout << "\nResultado: " << nitidez.nombre << "\n";
    imprimirImagen(salida, ancho, alto);

    // Ejecutar el filtro laplaciano.
    convolucionCPU(imagen, salida, ancho, alto, laplaciano);

    cout << "\nResultado: " << laplaciano.nombre << "\n";
    imprimirImagen(salida, ancho, alto);

    cout << "\nPrueba finalizada.\n";

    return 0;
}