## Adaptação e Avaliação da Rede CAS-Vit com Adapters

Este repositório documenta meu trabalho de adaptação da rede CAS-Vit para incluir adapters treináveis em cada bloco, deixando o backbone CAS-Vit congelado. Este projeto incluiu a construção de um dataset personalizado, adaptação do código original, experimentos de finetuning e avaliações em datasets diversos.

## Visão Geral do Projeto

A proposta foi explorar como a inclusão de adapters pode influenciar o desempenho da CAS-Vit em cenários de classificação, usando tanto um dataset customizado quanto o conjunto ImageNet-A para uma análise mais robusta. Abaixo, detalho as etapas que segui e os resultados obtidos.

## Estrutura

	1.	Preparação do Dataset
	2.	Adaptação do Código para Adapters
	3.	Avaliação Experimental
	4.	Resultados e Discussão

## Preparação do Dataset

Para desenvolver um conjunto de dados personalizado, capturei imagens de categorias de livre escolha usando meu celular. O processo incluiu:

	1.	Coleta de Imagens: Reuni pelo menos 200 imagens para cada classe. As classes foram selecionadas para fornecer variedade e contextos interessantes para os experimentos.
	2.	Divisão do Dataset: Organizei o dataset em proporções de 70% para treino, 15% para validação e 15% para teste. A estrutura final foi:

## Run in collab
```bash
!git clone https://github.com/Gregoriokn/CAS-ViT.git
```
```bash
!python classification/fine_tune.py  --data_path path_to_dataset --batch_size 32 --input_size 384 --finetune path_to_weights --lr 5e-5 --nb_classes n_classes  --model Name_model --device mps --adapters
```

# Project in progress 🚀