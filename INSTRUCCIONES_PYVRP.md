# Usar PyVRP para las instancias que aun no ganamos

El quickstart oficial del reto ([valeriaarciga/LRP_2026](https://github.com/valeriaarciga/LRP_2026))
envuelve **PyVRP** -- un solver de ruteo en C++ (Hybrid Genetic Search) --
adaptado para soportar costos y capacidades de depósito (lo que el PyVRP
normal de PyPI no soporta). En las instancias mock del propio repo, le gana al
baseline de Hexaly en 9 de 10 casos, ~2% en promedio. Es mucho más fuerte que
nuestro local search hecho a mano en Python.

Ya preparé en este proyecto:
- `solver/pyvrp_lrp/` -- el paquete `lrp` (lee/escribe el formato oficial
  SMIO directo, sin conversión).
- `solver/run_pyvrp.py` -- corre una instancia con una estrategia elegida y
  verifica con nuestro propio verificador oficial.
- `out/run_pyvrp_losing.sh` -- corre las 5 instancias que nos faltan ganar
  (small-03, small-08, medium-05, medium-07, medium-10) con 3 semillas cada
  una.

## 1. Instalar (una sola vez)

Primero revisa tu versión de Python -- el wheel precompilado que trae el repo
es para **Python 3.12 de 64 bits en Windows**:

```
python --version
```

Si dice 3.12.x, instala así:

```
pip install numpy tqdm
pip install "pyvrp @ https://github.com/valeriaarciga/LRP_2026/raw/main/pyvrp-0.13.3-cp312-cp312-win_amd64.whl"
```

Si tu Python NO es 3.12, dime qué versión tienes y buscamos otra forma
(crear un entorno virtual con 3.12, o compilar el fork desde su código
fuente -- más lento).

## 2. Probar rápido con una instancia

```
python solver/run_pyvrp.py Instancias_oficiales/clrp-small-08.txt out/pyvrp/small-08_test.sol.txt --strategy multi_start --time-limit 60 --seed 0 --verify
```

Debe imprimir `factible=True` y un costo, y al final el resultado del
verificador oficial. Si esto funciona, ya podemos correr el resto.

## 3. Correr las 5 instancias que nos faltan ganar

```
bash out/run_pyvrp_losing.sh
```

~55 minutos, 4 procesos a la vez, 3 semillas por instancia con estrategia
`multi_start`. Al final imprime los mejores costos logrados vs. lo que hay
que vencer en cada una.

## Estrategias disponibles (`--strategy`)

- `direct`: resuelve todo el LRP de una vez con PyVRP. Simple, buena para
  instancias chicas.
- `decomposition`: elige primero un subconjunto de depósitos (el más cercano
  a cada cliente, más los que hagan falta por capacidad, en orden de costo
  fijo creciente) y rutea solo esos. Pensada para instancias grandes.
- `multi_start`: prueba el subconjunto de `decomposition` más 4 variantes al
  azar con una corrida corta cada una, se queda con el mejor, y le da todo el
  tiempo restante a profundizar ahí. Es la opción por default en nuestros
  scripts.
