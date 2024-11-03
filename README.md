## Adaptação e Avaliação da Rede CAS-Vit com Adapters

Este repositório documenta meu trabalho de adaptação da rede CAS-Vit para incluir adapters treináveis em cada bloco, deixando o backbone CAS-Vit congelado. Este projeto incluiu a construção de um dataset personalizado, adaptação do código original, experimentos de finetuning e avaliações em datasets diversos.

## Visão Geral do Projeto

A proposta foi explorar como a inclusão de adapters pode influenciar o desempenho da CAS-Vit em cenários de classificação, usando tanto um dataset customizado quanto o conjunto ImageNet-A para uma análise mais robusta. Abaixo, detalho as etapas que segui e os resultados obtidos.

## Estrutura

	1.	Preparação do Dataset
	2.	Adaptação do Código para Adapters
	3.	Avaliação Experimental
    4.  Rodando no Colab

## Preparação do Dataset

Para desenvolver um conjunto de dados personalizado, capturei imagens de categorias de livre escolha usando meu celular. O processo incluiu:

	1.	Coleta de Imagens: Reuni pelo menos 200 imagens para cada classe. As classes foram selecionadas para fornecer variedade e contextos interessantes para os experimentos.
	2.	Divisão do Dataset: Organizei o dataset em proporções de 70% para treino, 15% para validação e 15% para teste. A estrutura final foi:

    dataset/
    ├── train/
    ├── val/
    └── test/

    3.	Pré-processamento: As imagens foram ajustadas para serem compatíveis com a entrada do modelo e padronizadas em tamanho e formato.

## Adaptação do Código para Adapters

O próximo passo foi adaptar o código da rede CAS-Vit para suportar adapters. Referenciei a implementação da arquitetura ViT com adapters, disponível aqui.

Principais Modificações

	•	Adição de Adapters: Incluí adapters em cada bloco da arquitetura CAS-Vit, permitindo que apenas eles fossem treináveis, mantendo o backbone CAS-Vit congelado.
	•	Ajustes no Forward: Alterei as funções forward_train e forward_test para incluir os adapters no fluxo de dados da rede.
	•	Parâmetros Congelados: Congelei todas as camadas da CAS-Vit, liberando apenas os adapters para treinamento, garantindo assim que as adaptações fossem isoladas.

## Avaliação Experimental

Com a rede adaptada e o dataset preparado, realizei duas avaliações distintas:

	1.	Finetuning com Dataset Personalizado
	•	CAS-Vit Original: Treinei a versão original da CAS-Vit com meu dataset, utilizando parâmetros de aprendizado adequados para finetuning.
	•	CAS-Vit com Adapters: Em paralelo, realizei o mesmo processo com a versão adaptada com adapters.
	2.	Avaliação nos Conjuntos de Teste
	•	Avaliação 1: Ambos os modelos foram avaliados no conjunto de teste do dataset personalizado. Utilizei a função classification_report da biblioteca scikit-learn para obter métricas detalhadas de desempenho por classe.
	•	Avaliação 2: Executei ambos os modelos no conjunto ImageNet-A para observar como a generalização do modelo CAS-Vit foi impactada pela adição dos adapters.


## Rodando no Colab
```bash
!git clone https://github.com/Gregoriokn/CAS-ViT.git
```
```bash
!python classification/fine_tune.py  --data_path path_to_dataset --batch_size 32 --input_size 384 --finetune path_to_weights --lr 5e-5 --nb_classes n_classes  --model Name_model --device mps --adapters
```