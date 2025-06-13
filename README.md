# Projeto Reconhecimento Facial 

# Descrição do pipeline

Utilizando a arquitetura YOLO versão 8, foi criado um modelo de visão computacional para detecção de rostos dos participantes do grupo e outras pessoas, como famosos atores, pessoas importantes do meio artístico, utilizando biblioteca ultralytics para realizar a maior parte do treinamento do modelo com dataset customizado, pegando imagens também da plataforma Kaggle datasets. 

# Arquitetura da rede utilizada

Utilizando arquitetura YOLOv11, processando o dataset de imagens contendo arquvos de test, train e valid contendo 228 imagens geradas com rotulagem personalizada dentro do ambiente de desenvolvimento roboflow, onde criamos o repositório com o dataset personalizado com os rostos dos participantes, contendo infinitas classes com o nome de cada participante e nome de celebridades, como por exemplo, nome do ator Alec Baldwin. 

Para a arquitetura YOLO personalizada, temos um arquivo yaml, onde contém as classes das imagens utilizadas no modelo, como Alec_Baldwin, Barbara, Vanessa e Giovana, para que o modelo possa indentificar cada participante, fazendo com que esta arquitetura identifique faces. 

Na implementação do modelo, utilizando arquitetura yolo v8, contém a biblioteca ultralytics, open cv, IPython.display.

# Estratégia de pré-processamento e treino

Com o dataset personalisado, utilizando a biblioteca from ultralytics import YOLO e from IPython.display import display, Image, contendo o prccesso de treinamento do dataset com as três pastas chamadas test, train e valid para a identificação dos rostos dos participantes e celebridades com a versão YOLOV8.  

# Resultados (acurácia, matriz de confusão, etc.)

Mostando a acurácia do modelo de Visão Computacional no final da implementação com as imagens rotuladas no arquivo modelo_yolo. 

# Dificuldades e aprendizados

Com a nova versão da YOLO, como adaptar para um dataset customizado e rotulado com as imagens geradas com a plataforma de visão computacional roboflow com o rosto dos participantes do projeto e celebridades das imagens do kaggle datasets. 
