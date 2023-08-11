# Projeto HUB IA || Turma 5

## Equipe

* **Laura Silva Lopes**
  
   [![GitHub](https://img.shields.io/badge/-GitHub-orange)](https://github.com/lauraslopes)
   [![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue)](https://www.linkedin.com/in/laura-silva-lopes/)

* **Tallyta Santos**

    [![GitHub](https://img.shields.io/badge/-GitHub-orange)](https://github.com/tallypy)
    [![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue)](https://www.linkedin.com/in/tallyta-santos/)

* **Tamires Brito**
     
    [![GitHub](https://img.shields.io/badge/-GitHub-orange)](https://github.com/tamiressbrito)
    [![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue)](https://www.linkedin.com/in/tamiresbrito/)

* **Vitória Marchesini**

    [![GitHub](https://img.shields.io/badge/-GitHub-orange)](https://github.com/MarchVitoria)
    [![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue)](https://www.linkedin.com/in/vit%C3%B3ria-marchesini-239412191/)


&nbsp;

## Índice

1. [Escopo do Projeto](#escopo-do-projeto)
2. [Resumo](#resumo)
3. [Motivação](#motivação)
4. [Objetivos](#objetivos)
5. [Detecção de Landmarks](#deteccao-landmarks)
6. [Dataset](#dataset)
7. [Arquivos e Pastas](#arquivos-e-pastas)
8. [Execução do Programa](#execução-do-algoritmo)

&nbsp;

## Resumo <a id="resumo"></a>

Este projeto consiste na aplicação prática de detecção de sonolência em tempo real usando visão computacional. A detecção de sonolência é uma técnica consolidada com um impacto significativo em áreas como gestão de frotas, contribuindo para evitar acidentes. O objetivo principal é treinar um modelo personalizado para a detecção de sonolência, aplicando uma metodologia bem definida.

&nbsp;

## Motivação <a id="motivação"></a>

Este projeto tem como objetivo criar uma aplicação prática que possa ser usada em apresentações, visitas e workshops, mostrando a capacidade do Hub em criar soluções úteis. Embora a metodologia seja estabelecida, o foco está no desenvolvimento de um modelo próprio de detecção de sonolência, diferenciando-o dos modelos já disponíveis. A motivação subjacente é impulsionada por uma preocupação fundamental: a segurança e bem-estar das pessoas. A detecção de sonolência em tempo real é um problema de extrema importância, não apenas para motoristas, mas também em diversas outras aplicações. No contexto de motoristas, a sonolência é uma das principais causas de acidentes de trânsito em todo o mundo. Além do trânsito rodoviário, a detecção de sonolência tem aplicações em diversas outras áreas, como a operação de maquinário pesado em setores industriais, e pode contribuir para a saúde individual, alertando sobre distúrbios do sono. Ao demonstrar uma solução tangível, estamos contribuindo para a educação pública e para a promoção de comportamentos mais seguros e saudáveis, tornando esta iniciativa um passo crucial em direção a um ambiente mais seguro e consciente.

&nbsp;

## Objetivos <a id="objetivos"></a>

* Aplicar a metodologia de detecção de sonolência.
* Treinar um modelo personalizado para caracterização da sonolência.
* Utilizar um dataset público.
* Parametrizar o modelo e realizar testes.
* Nomear o modelo treinado.

&nbsp;

## Detecção de Landmarks <a id="deteccao-landmarks"></a>

Para o treinamento do modelo detector de landmarks faciais, foi utilizado o algoritmo disponível em http://dlib.net/train_shape_predictor.py.html.

Os valores de parâmetros utilizados são listados a seguir e resultaram em uma taxa de erro médio de 5.49 no treinamento e 9.68 no teste.

Parâmetros:
* Cascade depth: 15
* Tree depth: 4
* 500 trees per cascade level.
* nu: 0.1
* oversampling amount: 5
* oversampling translation jitter: 0.1
* landmark_relative_padding_mode: 1
* feature pool size: 400
* feature pool region padding: 0
* 12 threads.
* lambda_param: 0.1
* 50 split tests.

**Cascade depth** e **tree depth** tiveram um impacto grande na acurácia do modelo, pois quanto maior, mais robusto fica. Porém, se esses valores não estiverem bem ajustados podem causar overfitting do modelo no dataset de treinamento e não generalização para o teste e inferência.

**Oversampling** se trata do processo de aumentar o número de exemplos no treinamento a partir dos dados que já existem. Para um dataset pequeno, esse número pode ser maior, porém, tem um enorme impacto no tempo de processamento. Para um oversampling amount de 300, o treinamento do modelo com o dataset utilizado levou 5 horas. Já, diminuindo esse valor para 5, o tempo de treinamento baixou para 8 minutos mas sem grande impacto na acurácia.

O número de **threads** leva em conta o número de núcleos (12) do processador utilizado para treinamento. Esse parâmetro impacta no tempo de treinamento mas não impacta em nada na acurácia.

Os demais parâmetros foram definidos após testes mas não tiveram impactos significativos na fase de treinamento.

&nbsp;

## Dataset <a id="dataset"></a>

O Dataset utilizado (**ibug_300W_large_face_landmark_dataset**) para o treinamento e teste deste projeto pode ser obtido em http://dlib.net/files/data/
e deve estar na pasta **data** para execução do algoritmo.

&nbsp;

## Arquivos e Pastas <a id="arquivos-e-pastas"></a>

O diretório do projeto está dividido da seguinte maneira:

* **data**: pasta que deve conter o dataset de treinamento e teste e o arquivo de áudio do sinalizador de sonolência.

* **model**: pasta que contém modelo de landmarks treinado;

* [main.py](main.py): arquivo principal para executar a detecção de sonolência em tempo real.

* [ear.py](ear.py): módulo para calcular o EAR (Eye Aspect Ratio) usado na detecção de sonolência.

* [mar.py](mar.py): módulo para calcular o MAR (Mouth Aspect Ratio) usado na detecção de bocejos.

* [moe.py](moe.py): módulo para calcular o MOE (Mouth Over Eye) usado na detecção de bocejos.

* [landmarks.py](landmarks.py): algoritmo de treinamento do modelo dlib de detecção de landmarks;

* [requirements.txt](requirements.txt): lista de bibliotecas requeridas para execução do projeto;

* [.gitignore](.gitignore): arquivo que contém os nomes dos arquivos e pastas que devem ser ignorados pelo *Git*;

* [README.md](README.md) : arquivo descrevendo detalhes do projeto.

&nbsp;

## Execução do Programa <a id="execução-do-programa"></a>

Para que o programa funcione, é fundamental instalar via pip todas as bibliotecas listadas no arquivo [requirements.txt](requirements.txt). Para isso, basta executar no terminal:

> pip install -r requirements.txt

Em seguida, executar o arquivo '[main.py](main.py)' para iniciar a detecção de sonolência em tempo real utilizando a webcam.

&nbsp;
