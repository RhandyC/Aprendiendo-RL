# Aprendiendo-RL
Este es un repositorio con codigos basicos en python explorando diferentes metodos orientado al RL

## 🔁 Value Iteration

**Value Iteration** es un algoritmo clásico de *Reinforcement Learning* (RL) usado para resolver procesos de decisión de Markov (MDP). Su objetivo es encontrar una política óptima mediante la actualización iterativa de los valores de los estados hasta que converjan.

### 🧠 ¿Cómo funciona?

1. **Inicialización**: Se asignan valores arbitrarios (por ejemplo, cero) a todos los estados.
2. **Iteración de valores**: Se actualiza cada valor de estado usando la ecuación de Bellman:

$V(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$

Donde:
- `V(s)` es el valor del estado `s`
- `P(s'|s,a)` es la probabilidad de transición
- `R(s,a,s')` es la recompensa al pasar de `s` a `s'` con acción `a`
- `γ` es el factor de descuento
3. **Convergencia**: Se repite el paso anterior hasta que el cambio en los valores sea menor que un umbral.
4. **Derivación de la política**: Con los valores estables, se elige la mejor acción para cada estado.

```bash
python -m venv venv
source venv/bin/activate  # Linux
pip install -r requirements.txt
```
