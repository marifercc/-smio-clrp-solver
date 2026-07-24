# Desafio de Localizacion y Ruteo SMIO-Hexaly -- Especificacion tecnica v1.0
(Texto extraido de hexaly_smio_spec_es.pdf, disponible en la Plataforma -> Reglamento)

## Formato de instancia (.txt)

```
# seed=<int>
# generator_version=<semver>
# osm_extract=<YYYY-MM-DD>   (solo FULL_MATRIX)
NAME : <instance_name>
CUSTOMERS : <n>
DEPOTS : <m>
VEHICLE_CAPACITY : <Q>
ROUTE_FIXED_COST : <g>
DISTANCE_FORMAT : COORDS | FULL_MATRIX
DEPOT_SECTION
<depot_id> <x> <y> <opening_cost> <capacity> <max_vehicles>
...
CUSTOMER_SECTION
<customer_id> <x> <y> <demand>
...
DISTANCE_SECTION      (solo si FULL_MATRIX; matriz (m+n)x(m+n), fila por fila)
EOF
```

- COORDS: distancia euclidiana redondeada a 1 decimal por par de nodos, ANTES de sumar (no se redondea la ruta total).
- FULL_MATRIX (instancias grandes, red vial real de Monterrey): puede ser asimetrica, no cumple designaldad triangular necesariamente.
- Indices en la matriz: depositos 1..m, luego clientes m+1..m+n.

## Formato de solucion (.sol)

```
# instance=<instance_name>
COST : <total_cost>
DEPOTS_OPENED : <n_open>
ROUTES : <n_routes>
DEPOT <depot_id>
ROUTE : <cust_1> <cust_2> ... <cust_last>
ROUTE : ...
DEPOT <depot_id>
ROUTE : ...
EOF
```

- Costo declarado debe coincidir con el recalculado (tolerancia 1e-4).
- Cada cliente exactamente una vez en total.
- No se permiten rutas vacias ni bloques DEPOT sin rutas.

## Funcion objetivo
min  Sum f_i*y_i  (apertura depositos)  +  g*|K|  (costo fijo por ruta)  +  Sum dist(R_k)  (distancia total)

## Restricciones
1. Cada cliente en exactamente una ruta.
2. Cada ruta opera desde un unico deposito abierto.
3. Demanda por ruta <= Q (capacidad vehiculo).
4. Demanda total asignada a deposito i <= W_i.
5. numero de rutas desde deposito i <= V_i (max. vehiculos).
6. Costo declarado == costo recalculado (tol. 1e-4).

## Escala de las 30 instancias oficiales
- Clientes: 200-3,000 / Depositos: 10-50
- Pequena/mediana: euclidianas sinteticas. Grande: red vial real ZMM (OSRM), FULL_MATRIX, asimetrica.
- Ejes de variacion: distribucion de clientes (uniforme/agrupada/mixta), posicion de depositos, distribucion de demanda, holgura de capacidad, holgura de limite de vehiculos, costo de ruta g.

## Puntuacion (ver reglamento paragrafo 8 para el detalle oficial)
- Delta_l,t = dias acumulados sosteniendo la BKS de la instancia l (medicion deslizante, precision de segundo).
- Bono B=3 dias equivalentes por sostener la BKS final de cada instancia.
- S_t = Sum Delta_l,t + B * Sum b_l,t
