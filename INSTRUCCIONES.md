# Como correr el solver en tu computadora

## 1. Requisitos
- Python 3.10 o superior instalado (verifica con: python --version)
- pip

## 2. Instalar dependencias
Abre una terminal DENTRO de esta carpeta y corre:

    pip install -r requirements.txt

## 3. Correr el solver sobre una instancia de practica

    python solver/clrp_solver.py instancias_practica/mock/mock_small_tight.txt salida.sol.txt --verify

Esto imprime el costo total y si el verificador oficial la marca como
factible (True/False). El archivo `salida.sol.txt` queda en el formato que
pide la Plataforma para enviar soluciones.

## 4. Correrlo sobre las instancias oficiales (cuando las descargues)
Copia el .txt de la instancia oficial a cualquier carpeta y apunta ahi:

    python solver/clrp_solver.py ruta/a/clrp-small-01.txt salida.sol.txt --verify

## 5. Estructura de este paquete

    solver/clrp_solver.py          -> el solver (heuristica constructiva)
    smio_clrp_verify_src/          -> parser + verificador oficial (de SMIO, sin modificar)
    instancias_practica/mock/      -> 10 instancias publicas para probar (con costo de referencia Hexaly)
    especificacion_tecnica.md      -> resumen del formato de archivo y las reglas de puntuacion
