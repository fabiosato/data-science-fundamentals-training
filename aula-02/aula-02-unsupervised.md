Fundamentos em Data Science
============================

## Aprendizado Não Supervisionado

Fábio Sato
fabiosato@gmail.com


---
# Aprendizado Não-Supervisionado

O objetivo é descobrir coisas interessantes sobre os dados
- Existe alguma forma de identificar padrões visualmente?
- Podemos descobrir subgrupos entre as variáveis ou observações?

Iremos abordar dois métodos principais:
- **Análise de Componentes Principais (PCA)**, uma ferramenta utilizada para visualização ou pré-processamento de dados antes da aplicação de uma técnica supervisionada
- **Clustering**, uma ampla família de algoritmos para descoberta de padrões desconhecidos em dados

---
# Aprendizado Não-Supervisionado - Desafios

Aprendizado não-supervisionado é mais subjetivo do que o aprendizado supervisionado, pois não há um objetivo claro para a análise
 
Entretanto, técnicas para aprendizado supervisionado estão crescendo em importância em vários campos

Exemplos:

- Identificação de grupos de consumidores caracterizados pelo seu padrão de navegação e de compras anteriores
- Agrupamento de filmes para recomendação

---
# Aprendizado Não-Supervisionado

É mais fácil obter dados não rotulados
   - medições de sensores
   - registros de logs de acesso
   - registros em tabelas de bancos de dados relacionais

Dados rotulados de foram geral requerem intervenção humana

Exemplo: É difícil de avaliar automaticamente de forma objetiva e geral pontuações de reviews de um filme: é favorável ou não?

---
# Análise de Componentes Principais (PCA)

PCA produz uma representação de menor dimensionalidade de um conjunto de dados

Além de servir como ferramenta para produzir variáveis derivadas para uso em aprendizado supervisionado, PCA também serve para visualização e compressão de dados

---
# Principal Compo

