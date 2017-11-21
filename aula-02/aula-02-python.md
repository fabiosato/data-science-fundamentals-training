Fundamentos em Data Science
============================

## Introdução ao Python - Tópicos Avançados

Fábio Sato
fabiosato@gmail.com


---
# Ordenação

Listas possuem um método `sort` que permite ordená-las

Caso seja necessário manter o valor original da lista, uma nova lista ordenada pode ser criada com a função `sorted`

```python
x = [4, 1, 2, 3]
y = sorted(x) # y = [1, 2, 3, 4], x não é modificado
x.sort() # x = [1, 2, 3, 4]
```

Para ordenar do maior para o menor, deve-se especificar o parâmetro `reverse=True` e a função `abs` para comparação dos valores

```python
x = sorted([-4, 1, -2, -3], key=abs, reverse=True)
```

---
# List Comprehensions

Frequentemente é necessário transformar uma lista em outra, selecionando apenas alguns elementos.

Um jeito "Pythônico" de realizar esta operação é através de *list comprehensions*

```python
even_numbers = [x for x in range(5) if x % 2 == 0] # [0, 2, 4]
squares = [x * x for x in range(5)] # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers] # [0, 4, 16]
```

---
# List Comprehensions

De forma similar listas podem ser transformadas em dicionários ou conjuntos

```python
square_dict = { x: x * x for x in range(5) }
square_set = { x * x for x in [1, -1] }
```

---
# List Comprehensions

Uma *list comprehension* pode utilizar múltiplos `for`:

```python
increasing_pairs = [(x, y)
         for x in range(10)
         for y in range(x + 1, 10)]
```

---
# Generators e Iterators

Um problema com listas é que elas podem facilmente crescer rapidamente.

```python
x = range(1000000) # não eficiente
```

Um *generator* é um recurso que permite iterar sobre uma lista, produzindo os valores de forma preguiçosa (*lazy*)

```python
def lazy_range(n):
    """Uma versão preguiçosa do range"""
	i = 0
    while i < n:
    	yield i
        i += 1

for i in lazy_range(10):
	do_something_with(i)
```

---
# Generators
Uma outra forma de criar generators é através do uso de parênteses