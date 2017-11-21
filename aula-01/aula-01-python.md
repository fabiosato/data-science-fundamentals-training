<!-- $theme: default -->

Fundamentos em Data Science
============================

## Introdução ao Python

Fábio Sato
fabiosato@gmail.com

---
# Introdução ao Python

- Python é uma linguagem de programação "interpretada", dinâmica, de código-fonte aberto e disponível para vários sistemas operacionais. 

- O código fonte é compilado para bytecode da máquina virtual Python

- O interpretador para Python é interativo, ou seja, é possível executá-lo sem fornecer um programa ou script para ele.

---
# Obtendo o Python

- O Python pode ser baixado do site http://www.python.org

- Entretanto, se você não possui ele instalado em seu sistema operacional, a distribuição Anaconda é amplamente utilizada para instalar e gerenciar os pacotes Python

- A última versão disponível atualmente é a 3.6. Recomendamos evitar utilizar versões 2.x e anteriores a 3.4.

- Se você não for utilizar o Anaconda, certifique-se de que possui o pacote `pip` instalado para gerenciar bibliotecas Python

- Também iremos utilizar o interpretador de comandos interativo chamdao `IPython`

---
# The Zen of Python

Python segue princípios Zen em seu projeto, que podem ser checados com o seguinte comando:

```python
import this
```

Um dos princípios mais discutidos é:

> There should be one  - and preferably only one - obvious way to do it

Código em Python escrito de acordo com este jeito "óbvio" (que pode não ser óbvio para um novato) é chamado de "Pythonic"

---
# Identação de código

Muitas linguagens utilizam chaves para delimitar blocos de código. Python utiliza identação:

```python
for i in [1, 2, 3, 4, 5]:
    print(i)
    for j in [1, 2, 3, 4, 5]:
        print(j)
        print(i+j)
print("Fim do Loop")
```

Isto torna o código Python bastante legível mas também requer que o programador seja cuidadoso com sua formatação

---
# Espaços em Branco

Espaços em branco são ignorados dentro de parênteses e colchetes e ajudam a escrever um código mais fácil de ler

```python
list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

easier_to_read_list_of_lists = [ [1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9] ]
```

---
# Módulos

Certos recursos do Python não são carregados por padrão

Isto ocorre para recursos incluídos na linguagem assim como recursos de terceiros

Para utilizar estes recursos é necessário importar os módulos que os contêm

```python
import re
# import re as regex (caso um outro módulo re j'
meu_regex = re.compile("[0-9]+", re.I)
```

`re`é um módulo contendo funções e constantes para manipular expressões regulares

depois  deste `import`as funções podem ser acessadas através do prefixo `re`

---
# Módulos

Se já existe um outro módulo `re` importado pode-se utilizar um `alias`:

```python
import re as regex
```

Esta forma de import é comumente utilizada para módulos com nome muito longo
```python
import matplotlib.pyplot as plt
```

---
# Módulos

Se apenas partes específicas de um módulo sejam desejadas pode-se importá-las diretamente e utilizá-las sem o uso do prefixo

```python
from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()
```

---
# Funções

Uma função é formada por um conjunto de instruções que recebem zero ou mais parâmetros de entrada e retornam um resultado de saída.

Em Python uma função é tipicamente criada com `def`

```python
def double(x):
	"""Aqui geralmente se escreve uma docstring que 
    explica o que a função faz.
    Por exemplo, esta função multiplica a entrada por 2"""
	
    return x * 2
```

---
# Funções
Funções em Python são cidadões de primeira classe.

Isto significa que pode-se atribuí-las a variáveis e fornecê-las como argumentos de outras funções

```python
def apply_to_one(f):
	"""chama a função f utilizando 1 como seu argumento"""
	return f(1)

my_double = double # referência a função definida anteriormente
x = apply_to_one(my_double) # qual é o resultado?
```

---
# Funções Lambda

Também é fácil criar funções curtas anônimas, chamadas `lambdas``

```python
y = apply_to_one(lambda x: x + 4) # resultado?
```

---
# Parâmetros de Funções

Os parâmetros de funções podem receber valores padrões

Isto permite especificar os valores dos parâmetros somente quando o valor desejado é diferente do padrão

```python
def my_print(message="minha mensagem"):
	print(message)
    
my_print("olá")
my_print()
```

Em algumas situações pode ser prático especificar os argumentos pelo nome:

```python
def subtract(a=0, b=0):
	return a - b
    
subtract(10, 5) # retorna 5
subtract(0, 5)  # retorna -5
subtract(b=5)   # também retorna -5
```

---
# Strings
Strings podem ser delimitadas por aspas simples ou duplas

```python
single_quoted_string = 'data science'
double_quoted_string = 'data science'
```

Python utiliza `\`para codificar caracteres especiais

```python
tab_string = '\t' # representa o caracter de tabulação
len(tab_string)   # 1
```

Se for necessário utilizar barras invertidas como caracteres (por exemplo para especificar um diretório do Windows) pode-se criar strings brutas (*raw*) com `r""`
```python
not_tab_string = r"\t"
len(not_tab_string) # 2
```

---
# Strings multi-linhas

Strings multi-linhas podem ser criados com três aspas duplas

```python
multi_line_string = """Esta é a primeira linha.
Esta é a segunda linha.
E esta é a terceira linha.
"""
```

---
# Exceções

Quando coisas erradas acontecem o Python lança uma excessão.

Excessões não tratadas irão causar a interrupção do programa em execução.

Para tratar/capturar exceções deve-se utilizar blocos `try`e `except`:

```python
try:
	print(0/0)

except ZeroDivisionError:
	print("Não pode ser dividido por zero")
```

---
# Listas

A estrutura de dados mais fundamental em Python é a lista.

A lista é uma simples coleção ordenada, similar a uma matriz/vetor, porém com funcionalidades extras.

```python
integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [ integer_list, heterogeneous_list]

list_length = len(integer_list) # 3
list_sum = sum(integer_list) # 6
```

---
# Acesso a Listas

O n-ésimo elemento de uma lista pode ser acessado com colchetes

Índices são baseados em zero

```python
x = range(10) # igual a lista [0, 1, ..., 9]
zero = x[0]
one = x[1]
nine = x[-1]  # modo pythônico para acessar o último elemento
eight = x[-2] # modo pythonico para acessar o penúltimo elemento
x[0] = -1 # x agora é [-1, 1, 2, 3, ..., 9]
```

---
# Fatiando listas

Os colchetes também podem ser usados para "fatiar" listas

```python
first_three = x[:3] # [-1, 1, 2]
three_to_end = x[3:] # [3, 4, ..., 9]
one_to_four = x[1:5] # [1, 2, 3, 4]
last_three = x[-3:] # [7, 8, 9]
without_first_and_last = x[1:-1] # [1, 2, ..., 8]
copy_of_x = x[:] # [-1, 1, 2, ..., 9]

```

---
# List Membership

```python
1 in [1, 2, 3] # True
0 in [1, 2, 3] # False
```

---
# Concatenando Listas

Listas são mutáveis

```python
x = [1, 2, 3]
x.extend([4, 5, 6]) # x = [1, 2, 3, 4, 5, 6]
````

Caso não se deseja alterar uma lista, pode-se utilizar o operador de adição para criar uma nova

```python
x = [1, 2, 3]
y = x + [4, 5, 6] # y = [1, 2, 3, 4, 5, 6] e x = [1, 2, 3]
```

O método `append`deve ser utilizar para concatenar somente um elemento

```python
x = [1, 2, 3]
x.append(0)
y = x[-1]  # y = 0
z = len(x) # z = 4
```

---
# Descompactando/Desempacotando (*Unpack*) Listas 

Pode-se designar o valor de elementos de uma lista diretamente a variáveis.
```python
x, y = [1, 2]  # x = 1 e y = 2
```

Caso um dos elementos não seja desejado, deve-se utilizar o `_` durante o *unpacking*

```python
_, y = [1, 2] # y = 2
```

---
# Tuplas

Tuplas são listas imutáveis. Qualquer operação suportada por uma lista que não altere o seu conteúdo pode ser realizada por uma tupla.

Tuplas são criadas com parênteses (ou nada) ao invés de colchetes.

```python
my_list = [1, 2]
my_tuple = (1, 2)
other_tuple = 3, 4
my_list[1] = 3

try:
    my_tuple[1] = 3
except TypeError:
    print("Cannot modify a tuple")
```

---
# Tuplas - Resultados de Funções

Tuplas são úteis para retornar múltiplos valores a partir de funções

```python
def sum_and_product(x, y):
    return (x + y), (x * y)
    
sp = sum_and_product(2, 3) # sp = (5, 6)
s, p = sum_and_product(5, 10) # s = 15 e p = 50
```

---
# Tuplas - Atribuição a Variáveis

```python
x, y = 1, 2 # x = 1 e y = 2
x, y = y, x # jeito pythonico de fazer swap de variáveis
```

---
# Dicionários

Dicionários associam valores e chaves: $k \rightarrow v$

Permitem a busca rápida de valores pela chave

```python
dict_vazio = {} # jeito pythonico de criar um dicionario
dict_vazio2 = dict() # não-pythonico
notas = {"João": 80, "Maria": 100} # literal para dicionário
```

---
# Busca em Dicionários

Um valor pode ser buscado através da chave através de colchetes

```python
notas_joao = notas["João"]
```

Uma exceção será lançada caso a chave não exista no dicionário

```python
try:
    notas_manoel = notas["Manoel"]
except KeyError:
    print("A chave não está presente neste dicionário")
```

Para verificar a presença de uma chave deve-se utilizar `in`:

```python
joao_tem_nota = "João" in notas  # True
manoel_tem_nota = "Manoel" in notas # False
```

---
# Dicionários - Retorno Padrão

Dicionários possuem um método `get` que retorna um valor padrão quando a chave não está presente no dicionário

```python
nota_joao = notas.get("João", 0) # nota_joao = 80
nota_manoel = notas.get("Manoel", 0) # nota_manoel = 0
nota_ninguem = notas.get("ninguém") # nota_ninguem = None
```

---
# Dicionários - Novas Entradas

Novas entradas podem ser adicionadas com colchetes

```python
notas["Maria"] = 99  # substitui o valor anterior
notas["Silvia"] = 100 # adiciona uma nova entrada
num_estudantes = len(notas)  # 3
```

---
# Dicionários - Dados Estruturados

Dicionários são frequentemente utilizados para representar dados estruturados

```python
tweet = {
    "user": "joao",
    "text": "Data Science",
    "retweet_count": 1,
    "hashtags": ["#data", "#science"]
}
```

---
# Dicionários - Mais Busca

Além de busca pelas chaves, podemos verificar todas elas

```python
tweet_keys = tweet.keys()      # lista de chaves
tweet_values = tweet.values()  # lista de valores
tweet_items = tweet.items()    # lista de tuplas (chave, valor)

"user" in tweet_keys # True, mas lento
"user" in tweet      # jeito pythonico
```

---
# Sets

Outra estrutura de dados é `set` que representa um conjunto de elementos **distintos**

Conjuntos são eficientes. Para coleções grandes o teste de presença (*membership*) em conjuntos é mais rápido do que em listas/tuplas/dicionários.

```python
stopwords = ["a", "an"] + hundreds_of_words + ["yet" + "you"]

"zip" in stopwords # False, tem que verificar cada elemento

stopwords_set = set(stopwords)

"zip" in stopwords_set # muito mais rápido
```

---
# Sets - Dados Distintos/Únicos

Sets são úteis para encontrar itens únicos de uma lista

```python
item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list)  # 6
item_set = set(item_list) # {1, 2, 3}
num_distinct_items = len(item_set) # 3
distinct_item_list = list(item_set) # [1, 2, 3]
```

---
# Controle de Fluxo

Como na maioria das linguagens de programação, operações condicionais são definidas com `if`:

```python
if 1 > 2:
    message = "if only 1 were greater than two"
elif 1 > 3:
    message = "elif stands for 'else if'"
else:
    message = "when all else fails use else (if you want to)"
```

---
# Operador ternário

O operador ternário oferece uma maneira rápida de escrever uma condicional

```python
parity = "even" if x % 2 == 0 else "odd"
```

---
# Laços - While

Laços podem ser definidos com blocos `while`

```python
x = 0
while x < 10:
    print(x, " is less than 10")
    x += 1
```

---
# Laços - For
Laços são mais comumentes escritos com `for``

```python
for x in range(10):
    print(x, " is less than 10")
```

---
# Laços - Continue e Break

Lógicas mais complicadas podem ser escritas com `continue` e `break`

```python
for x in range(10):
    if x == 3:
        continue # pula para a próxima iteração
    if x == 5:
        break  # termina o laço imediatamente
    print(x)
```

---
# Booleans

Booleans em Python funcionam como na maioria das outras linguages, com a diferença de que iniciam com uma letra maiúscula

```python
one_is_less_than_two = 1 < 2 # True
true_equals_false = True == False # False
```

---
# None

Python utiliza `None`para indicar um valor não existente. Ele é semelhante ao `null`de outras linguagens

```python
x = None
print(x == None) # True - jeito não-pythonico
print(x is None) # True - jeito pythonico
```

---
# Exercícios

1 - Escreva uma função que ordene uma lista (sem utilizar funções/métodos de ordenação da API do Python)

2 - Escreva uma função que implemente as seguintes equações, recebendo uma lista para parâmetro:

$$ \overline x = \frac {\sum _{i=1}^{n}{x_i}}{n}$$

$$ \sigma = \sqrt {\frac {\sum _{i=1}^{n}{(x_i - \overline x)^2}}{n - 1}} $$