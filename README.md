# wine_quality

O teste aborda a predição da qualidade de vinhos com base nas métricas físico-químicas e a detecção de alguns insumos. O data set contém 6497 amostras etiquetadas com níveis de qualidade que variam entre 3 e 9.
Consistência dos dados:
- Os dados contêm uma variável categórica chamada “type” que pode assumir os valores “White” e “Red”. Estes valores foram padronizados em 1 e 2, respectivamente.
- Na conversão dos dados foram encontrados alguns valores discrepantes (possivelmente erros de captação) na coluna “alcohol” como 116.333.333.333.333. Após análise de a quais classes estes valores pertenciam e o impacto da remoção do exemplo completo no balanceamento de classes, estas linhas foram tratadas como outliars e estes valores foram removidos do conjunto.
- Na análise de balanceamento de classes foi detectado um grande desbalanceamento de exemplos mostrado na tabela a seguir:
Qualidade	Nº de exemplos
6	        2815
5	        2129
7	        1069
4	        216
8	        193
3	        30
9	        5

Para endereçar o desbalanceamento de classes foram adotadas duas estratégias: 
1) A primeira foi incluir nos testes classificadores que não sofram tanta influência deste fator como: Árvores de decisão e Random Forest. 2) A segunda foi utilizar uma função de custo que observe a quantidade de acertos em cada classe, para que os erros acertos em classes com menos exemplos (9,3,8 e 4) também exerçam influência significativa nos resultados. A métrica utilizada foi uma adaptação do F1 Score para avaliação de classificação multi-classe.

- O melhor classificador foi escolhido através de uma busca GridSearch, que testou (Arvore de decisão, Random Forest, K-Vizinhos mais próximos e Multi-Layer Perceptron). A técnica que apresentou o melhor F1 score foi a Random Forest, gerando um F1 Score de aproximadamente 70%.
- Para se avaliar a importância das variáveis foi realizada uma busca combinatória envolvendo todas as características e as possíveis combinações entre elas. Através destes experimentos foi possível identificar que todas as caraterísticas são importantes na classificação. A melhor F1 score foi obtida em uma configuração envolvendo todas as características.
