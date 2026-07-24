# SMIO-Hexaly CLRP Solver

Heuristica para el **SMIO-Hexaly Location Routing Challenge 2026**: un problema
de localizacion y ruteo capacitado (CLRP). Hay que decidir que depositos
abrir, que clientes asigna cada uno, y como armar las rutas de vehiculos,
minimizando costo de apertura + costo fijo por ruta + distancia total.

## Requisitos

- Python 3.10 o superior
- El parser + verificador oficial del challenge (`smio_clrp_verify`), provisto
  por la organizacion en la Plataforma. **No esta incluido en este repo**
  (ver mas abajo) -- hay que descargarlo aparte y copiarlo en
  `smio_clrp_verify_src/` en la raiz de este proyecto.

```
pip install -r requirements.txt
```

## Uso

```
python solver/clrp_solver.py <instancia.txt> <salida.sol.txt> --verify --time-limit 20
```

- `instancia.txt`: instancia de entrada (ver `instancias_practica/mock/` para
  10 instancias publicas de prueba)
- `salida.sol.txt`: donde se escribe la solucion, en el formato oficial
- `--verify`: corre el verificador oficial sobre la solucion generada y
  muestra si es factible
- `--time-limit`: segundos de busqueda local despues de la construccion
  inicial (default 20, poner 0 para desactivarla)

## Pipeline

1. **Construccion golosa** -- localizacion de depositos capacitada (greedy
   add), reparacion de bin-packing, First-Fit-Decreasing para armar rutas, y
   nearest-neighbour + 2-opt para ordenar cada una.
2. **Busqueda local** -- tres jugadas que se alternan hasta converger:
   relocate (mover un cliente de ruta), swap (intercambiar dos clientes), y
   cierre de deposito (apagar un deposito entero si ya no compensa su costo
   de apertura).
3. **LNS (Large Neighborhood Search)** -- una vez agotadas esas jugadas, si
   sobra tiempo: destruye a proposito una porcion de la solucion (clientes al
   azar, por zona geografica, o rutas completas) y la reconstruye greedy,
   repitiendo para escapar de optimos locales. Se guarda siempre la mejor
   solucion vista.

## Estructura

```
solver/clrp_solver.py       -> el solver
instancias_practica/mock/   -> 10 instancias publicas de practica
especificacion_tecnica.md   -> formato de archivo y reglas de puntuacion
INSTRUCCIONES.md            -> como correrlo paso a paso
smio_clrp_verify_src/       -> (no incluido) verificador oficial -- descargar
                                de la Plataforma y copiar aca
```

## Resultados sobre las instancias de practica

Con `--time-limit 20`, las 10 instancias dan `feasible=True`:

| Instancia | Costo |
|---|---|
| mock_small_consolidation | 18,122.96 |
| mock_small_loose_clustered | 23,837.31 |
| mock_small_loose | 29,322.47 |
| mock_small_moderate | 40,386.87 |
| mock_small_asym_depot_binding | 37,966.22 |
| mock_small_asym_vehicle_binding | 41,792.00 |
| mock_small_tight | 42,022.34 |
| mock_medium_loose | 57,562.56 |
| mock_medium_tight | 68,553.04 |
| mock_medium_moderate | 74,163.58 |
