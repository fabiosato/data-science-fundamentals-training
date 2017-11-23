# Respostas "Pythonicas" aos Exercícios :smile:

1 - Escreva uma função que ordene uma lista (sem utilizar funções/métodos de ordenação da API do Python)

```python

def quicksort(v):
    if len(v) <= 1: 
        return v
    
    pivot = v[0]
    equal = [x for x in v if x == pivot]
    lesser = [x for x in v if x < pivot]
    greater = [x for x in v if x > pivot]
    return quicksort(lesser) + equal + quicksort(greater)
```

---
2 - Escreva uma função que implemente as seguintes equações, recebendo uma lista para parâmetro:

$$ \overline x = \frac {\sum _{i=1}^{n}{x_i}}{n}$$

```python
def avg(v):
	return reduce(lambda x, y: x + y, v) / len(v)
```

$$ \sigma = \sqrt {\frac {\sum _{i=1}^{n}{(x_i - \overline x)^2}}{n - 1}} $$

```python
import math

def stdev(v):
    mean = avg(v)
    sum_diff = sum((x - mean)**2 for x in v)
    return math.sqrt(sum_diff/(len(v)-1))
```

---
# Revisão de Conceitos - Discussão 

Como você resolveria os seguintes problemas com Machine Learning?


| Problema           | Dados     | Categoria/Tipo   | Critério/Otimização    |
|--------------------|-----------|------------------|------------------------|
| Diagnóstico Médico |           |                  |                        |
| Veículo Autônomo   |           |                  |                        |
| Falha Mecânica     |           |                  |                        |
| Consumo de Energia |           |                  |                        |
| Bolsa de Valores   |           |                  |                        |
| Jogo de Xadrez     |           |                  |                        |                        | Chatbot            |           |                  |                        |