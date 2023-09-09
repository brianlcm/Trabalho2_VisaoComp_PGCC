# Segundo trabalho de implementação da disciplina de Visão Computacional do PPGCC/UFJF

O trabalho consiste na implementação de redes neurais convolucionais clássicas da literatura:

• LeNet

• AlexNet

• VGG

As redes neurais implementadas foram avaliadas nos seguintes conjuntos de dados:

• Fashion-MNIST

• CIFAR-10

Para iniciar o treinamento, basta executar o arquivo train.py. Os parâmetros, rede e dataset utilizada podem ser modificados entre as linhas 30 e 37 desse arquivo.

Para avaliar o modelo treinado, basta executar o arquivo evaluate.py utilizando os mesmos hiperparâmetros do treinamento.

O modelo, gráficos de acurácia e perda são salvos na pasta "experiments" após o término do treinamento. Além disso, após executar o arquivo evaluate.py, a matriz de confusão e um arquivo .csv com algumas métricas também são
salvos na pasta "experiments".
