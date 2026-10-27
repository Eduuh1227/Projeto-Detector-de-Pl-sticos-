# Aplicação de Processamento Digital de Imagens e Visão Computacional para Identificação de Resíduos Plásticos em Praias e Rios

**APS — Processamento de Imagens e Visão Computacional**

Aplicação acadêmica em Python para carregar fotografias, comparar técnicas de processamento de imagens e integrar um detector YOLO de resíduos plásticos visíveis.

**Estado da entrega:** a infraestrutura de processamento, interface, treinamento, avaliação e testes está implementada. O projeto é entregue **sem dataset, sem pesos treinados para plástico e sem métricas de desempenho**. O detector ainda precisa ser treinado e avaliado com dados reais. A aplicação abre e permite explorar o processamento mesmo nessa condição.

## Contexto, problema e objetivo

O projeto investiga como técnicas de processamento digital de imagens e visão computacional podem apoiar a localização de resíduos plásticos visualmente perceptíveis em fotografias de praias e margens de rios. Esses ambientes apresentam areia, vegetação, água, sombras, reflexos e objetos de materiais diferentes, o que torna insuficiente identificar resíduos apenas pela intensidade ou pela cor dos pixels.

O objetivo é construir um fluxo reproduzível: entrada da fotografia, validação, preparação, comparação de filtros e segmentação tradicional, seguida de detecção por um modelo treinado e apresentação das caixas delimitadoras. O escopo é a análise de fotografias convencionais. **Não há detecção de microplásticos, análise química do material ou estimativa automática de impacto ambiental.**

Nenhum filtro é declarado superior sem um experimento que sustente essa conclusão. A limiarização é uma demonstração acadêmica e não determina que uma região seja plástico.

## Equipe e divisão do trabalho

| Integrante | Responsabilidade principal | Módulos de referência |
| --- | --- | --- |
| Eduardo | Processamento das imagens e integração do sistema | `processamento/`, `app.py` |
| Ryan | Organização e anotação do dataset, configuração, treinamento e avaliação do YOLO | `dataset/`, `modelo/dados.py`, `modelo/treinamento.py`, `modelo/avaliacao.py` |
| Bárbara | Interface, testes de utilização e apresentação dos resultados | `interface/`, `tests/`, documentação |

As funções de processamento não dependem do Streamlit nem de um modelo treinado. O detector não depende da interface. A aplicação conecta esses módulos por contratos explícitos, permitindo trabalho independente e revisão conjunta da integração.

## Tecnologias

| Dependência | Uso no projeto |
| --- | --- |
| Python 3 | Aplicação e scripts; Python **3.12** recomendado para o ambiente do projeto |
| OpenCV | Decodificação, cores, redimensionamento, filtros, segmentação, desenho e codificação PNG |
| NumPy | Representação e operações sobre os pixels |
| Matplotlib | Histograma opcional das intensidades de cinza |
| Ultralytics YOLO | Inferência, fine-tuning e avaliação de detecção de objetos |
| Streamlit | Upload, controles, apresentação e exportação na interface |
| PyYAML | Leitura segura e escrita da configuração do dataset |
| pytest | Testes de processamento, contratos, validação e utilização |

PyTorch e outras bibliotecas necessárias ao Ultralytics/Streamlit são resolvidas como dependências transitivas durante a instalação. O arquivo `requirements.txt` lista as dependências utilizadas diretamente pelo projeto.

## Estrutura

```text
aps-plasticos/
├── app.py
├── requirements.txt
├── Dockerfile.vercel
├── .dockerignore
├── vercel.json
├── pytest.ini
├── README.md
├── VALIDACAO.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── processamento/
│   ├── __init__.py
│   ├── preprocessamento.py
│   ├── filtros.py
│   └── segmentacao.py
├── modelo/
│   ├── __init__.py
│   ├── dados.py
│   ├── detector.py
│   ├── treinamento.py
│   ├── avaliacao.py
│   └── resultados.py
├── interface/
│   ├── __init__.py
│   ├── componentes.py
│   └── deteccao.py
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── plastico.yaml
├── modelos/
│   └── .gitkeep
├── resultados/
│   ├── filtros/
│   ├── deteccoes/
│   └── metricas/
└── tests/
    ├── conftest.py
    ├── test_preprocessamento.py
    ├── test_detector.py
    ├── test_resultados.py
    ├── test_interface.py
    ├── test_interface_deteccao.py
    ├── test_dados.py
    └── test_treinamento_avaliacao.py
```

Os diretórios vazios são preservados com `.gitkeep`. `resultados/treinamentos/` e os diretórios de cada execução são criados quando necessários. A pasta opcional `modelos/base/` pode guardar os pesos pré-treinados usados exclusivamente para iniciar o treinamento.

## Instalação e execução

Execute os comandos a partir da pasta `aps-plasticos`, onde está `app.py`. É necessário ter o Python instalado e disponível no terminal. A instalação das dependências precisa de acesso à internet e pode baixar bibliotecas grandes, especialmente o PyTorch.

### Windows — PowerShell

```powershell
cd aps-plasticos
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Se o PowerShell bloquear o script de ativação, não é necessário alterar a política do sistema. Utilize diretamente o Python do ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Se o computador usa o inicializador `py`, a criação do ambiente pode ser feita com `py -3.12 -m venv .venv`.

### Windows — Prompt de Comando (CMD)

```bat
cd aps-plasticos
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
streamlit run app.py
```

### Linux ou macOS

```bash
cd aps-plasticos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abra no navegador o endereço local informado pelo Streamlit. Encerre o servidor com `Ctrl+C` no terminal. Para executar sem abrir automaticamente o navegador, acrescente `--server.headless true`.

Os exemplos seguintes pressupõem o ambiente virtual ativo. No Windows, também é possível substituir `python` por `.\.venv\Scripts\python.exe` em qualquer comando.

## Uso da aplicação

1. **Upload:** selecione um JPG, JPEG ou PNG. O conteúdo é decodificado e validado; apenas renomear um arquivo para `.png` não o torna uma imagem válida. O limite é de 20 MB e 24 megapixels. A fotografia original e suas dimensões são mostradas. PNG com transparência é composto sobre fundo branco.
2. **Processamento:** compare a imagem de referência, o filtro Gaussiano e o filtro de mediana. Altere o tamanho do kernel. Explore a escala de cinza, a limiarização manual ou Otsu e as operações de abertura e fechamento. O histograma de intensidades pode ser ativado.
3. **Detecção:** se não houver pesos personalizados, a interface informa que o modelo ainda não foi treinado. Se houver um arquivo `.pt` na pasta `modelos/`, selecione-o, ajuste os parâmetros e clique em **Detectar resíduos**. O carregamento verifica a compatibilidade das classes antes da inferência.
4. **Informações acadêmicas:** consulte a distinção entre processamento e detecção e o escopo do sistema.

A numeração acompanha as etapas visíveis: antes do upload, aparecem Upload (1), Detecção (2) e Informações acadêmicas (3). Após enviar uma imagem válida, Processamento é inserido como etapa 2, seguido por Detecção (3) e Informações acadêmicas (4).

A comparação de processamento usa uma versão proporcional com até **1600 pixels no maior lado**, sem ampliar imagens menores. A inferência recebe a imagem original decodificada, sem os filtros comparativos. Por isso, as caixas retornadas correspondem aos pixels da imagem original, e não à resolução de exibição.

Alterar a fotografia, o modelo ou os parâmetros de inferência invalida o resultado anterior na sessão. Os filtros podem ser explorados separadamente, pois não são a entrada do detector. A interface usa CPU por padrão; treino e avaliação permitem selecionar outro dispositivo pela linha de comando.

Uma inferência real que não retorna objetos apresenta contagem zero. Isso não comprova ausência de plástico na fotografia. A confiança apresentada para cada caixa é o escore do modelo e não deve ser chamada de acurácia ou probabilidade calibrada de acerto.

## Hospedagem no Vercel

O projeto inclui `Dockerfile.vercel` e `vercel.json` para executar a mesma interface Streamlit com o **Container Runtime do Vercel**, atualmente em Beta. O container instala as bibliotecas de sistema necessárias ao OpenCV e o PyTorch para CPU. Ele escuta na porta 80, ou na variável `PORT` quando configurada. [Documentação do Container Runtime](https://vercel.com/docs/functions/runtimes/container).

Na importação do repositório no Vercel, selecione a raiz do projeto e o framework **Container**. Não configure uma pasta de saída estática nem use `streamlit run` como comando de build: o processo é iniciado pelo `CMD` do Dockerfile. A publicação usa os arquivos da branch selecionada; datasets, ambientes virtuais e resultados locais ficam fora da imagem de implantação.

O modo hospedado (`APS_HOSPEDADO=1`, configurado no Dockerfile) mantém upload, filtros, segmentação e a mensagem de ausência de modelo. Existem diferenças em relação à execução local:

- **Upload de até 4 MB:** limite conservador abaixo do teto HTTP de 4,5 MB do Vercel. A execução local continua aceitando até 20 MB. [Limites de Functions](https://vercel.com/docs/functions/limitations).
- **Prévias leves:** as imagens coloridas de exibição são limitadas a 1280 pixels por lado e transmitidas em JPEG; os arrays da análise e os arquivos exportados preservam seus próprios dados e resoluções.
- **Resultados por download:** quando houver inferência real, a exportação completa é um ZIP montado em memória, sem prometer armazenamento permanente no servidor. Downloads acima de 4 MB são bloqueados com orientação para reduzir a resolução ou usar a versão local; nenhum arquivo é truncado.
- **Sessão temporária:** reinícios, novas implantações ou reconexões podem exigir novo upload. WebSockets são suportados, mas a conexão termina ao atingir a duração máxima da função; uma nova conexão pode usar outra instância. [WebSockets no Vercel](https://vercel.com/docs/functions/websockets).
- **Pesos locais ao container:** não há download automático de modelo. Para habilitar detecção futuramente, inclua o checkpoint compatível no processo de implantação e avalie-o antes de apresentar desempenho. O `.gitignore` continua evitando a publicação acidental de pesos pesados.

Treinamento e avaliação devem ser executados no ambiente de desenvolvimento; não são disparados por visitantes do site. A publicação não substitui o treinamento pendente nem produz métricas científicas.

## Arquitetura e contratos dos módulos

| Módulo | Entrada e saída principais |
| --- | --- |
| `preprocessamento.py` | Lê caminho, bytes ou arquivo binário; retorna NumPy `uint8`, formato HWC, três canais BGR. Valida, converte cores e redimensiona. |
| `filtros.py` | Recebe uma imagem e retorna outra imagem filtrada, sem modificar a original. Kernels devem ser ímpares e pelo menos 3. |
| `segmentacao.py` | Recebe BGR ou cinza; retorna máscara binária e o limiar efetivamente usado. |
| `dados.py` | Valida YAML, imagens, labels e partições; produz manifesto e hashes de proveniência. |
| `detector.py` | Carrega pesos locais, verifica classes, executa YOLO e retorna `ResultadoDeteccao` com imagem BGR, lista de `Deteccao` e metadados. |
| `treinamento.py` | Valida dados e parâmetros, realiza fine-tuning e retorna o caminho exclusivo do melhor checkpoint copiado. |
| `avaliacao.py` | Executa avaliação do split solicitado e salva somente métricas válidas retornadas pelo YOLO. |
| `resultados.py` | Salva imagens e JSON de uma inferência, com identificadores exclusivos. |
| `interface/` | Apresenta controles e resultados; não implementa operações de treinamento. |

O contrato BGR segue a entrada NumPy documentada pelo Ultralytics; a conversão para RGB ocorre na apresentação. Normalização e redimensionamento internos do detector são feitos pelo próprio YOLO. A função didática `padronizar_resolucao` acrescenta bordas para obter dimensões exatas, mas não é aplicada antes da inferência. Consulte a [documentação oficial de predição](https://docs.ultralytics.com/modes/predict/) para o formato dos resultados.

Cada detecção contém `classe_id`, `classe`, `confianca` e `caixa_xyxy`. As coordenadas são contínuas em pixels: `(x1, y1, x2, y2)`. A renderização com OpenCV acontece sobre uma cópia da imagem.

## Preparação do dataset

O projeto não fornece fotografias, anotações ou divisão pronta de dados. A equipe deverá coletar ou obter imagens reais cujo uso esteja autorizado, registrar origem e licença e definir uma política de anotação antes do treinamento.

### Organização e cuidados experimentais

- Use fotografias representativas de praias e margens de rios, incluindo variações de iluminação, distância, escala, oclusão e fundo.
- Defina o que será marcado como plástico visível e como lidar com materiais ambíguos. A fotografia não confirma composição química; casos duvidosos devem ser revisados com um critério consistente.
- Anote todos os objetos da classe que atendam ao protocolo, usando caixas delimitadoras. Revise as anotações, preferencialmente por uma segunda pessoa.
- Inclua imagens negativas revisadas, com ausência de objetos da classe segundo o protocolo. Elas recebem um `.txt` vazio correspondente, não um label ausente.
- Separe `train`, `val` e `test` por grupos de local, sessão de captura ou sequência, sempre que possível. Fotografias quase iguais, recortes e versões editadas da mesma cena não devem cair em partições diferentes.
- Use `train` para aprender os parâmetros, `val` para escolher decisões durante o desenvolvimento e `test` somente para a avaliação final. Não ajuste parâmetros com base no resultado do teste.
- Registre critérios de inclusão, tamanho de cada partição, quantidade de objetos e de negativos, condições de coleta e decisões de anotação quando os dados existirem. Não preencha esses registros com números estimados como se fossem observados.

Coloque cada imagem em `dataset/images/<split>/` e seu label de mesmo nome-base em `dataset/labels/<split>/`. Subpastas são aceitas quando espelhadas entre `images` e `labels`. Evite nomes-base repetidos, mesmo com extensões distintas.

### Configuração YAML

O arquivo fornecido é:

```yaml
path: .
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: plastico
```

Os scripts do projeto resolvem `path` em relação ao próprio arquivo YAML. O YAML original permanece portátil e sem caminhos absolutos fixos; uma cópia `dataset.runtime.yaml` com os caminhos resolvidos é criada na pasta de cada execução. Use os scripts deste projeto para preservar esse comportamento.

Esta implementação aceita diretórios locais na estrutura `images/<partição>` e `labels/<partição>`. Não aceita URLs, listas de arquivos ou chaves de download/script no YAML. É uma escolha de escopo mais restrita que as possibilidades gerais do formato Ultralytics.

### Formato das anotações YOLO

Cada linha de um `.txt` representa uma caixa no formato:

```text
<classe_id> <centro_x_normalizado> <centro_y_normalizado> <largura_normalizada> <altura_normalizada>
```

Esse texto é apenas um esquema, não uma anotação válida para copiar no dataset. No MVP, `classe_id` é o inteiro `0`. As demais informações devem ser calculadas a partir das caixas desenhadas na fotografia real.

Para uma imagem de largura `W`, altura `H` e caixa em pixels `(x1, y1, x2, y2)`:

```text
centro_x_normalizado = (x1 + x2) / (2 * W)
centro_y_normalizado = (y1 + y2) / (2 * H)
largura_normalizada  = (x2 - x1) / W
altura_normalizada   = (y2 - y1) / H
```

As coordenadas devem ser finitas e normalizadas entre 0 e 1; largura e altura devem ser positivas, e a caixa deve permanecer dentro da imagem. O formato é descrito na [documentação de datasets de detecção do Ultralytics](https://docs.ultralytics.com/datasets/detect/).

O validador verifica decodificação das imagens, correspondência de labels, cinco campos por linha, classes e coordenadas, labels órfãos, colisão de nomes e duplicatas exatas entre partições usando SHA-256. A checagem de hash não encontra automaticamente fotos quase iguais nem garante que as caixas humanas estejam corretas.

Para verificar a estrutura sem treinar, depois de adicionar dados reais:

```bash
python -c "from modelo.dados import validar_dataset; print(validar_dataset().contagens)"
```

Por padrão, essa chamada exige `train` e `val` com pelo menos uma anotação positiva em cada um. Todos os splits preenchidos são examinados. Dataset vazio gera uma mensagem de erro; nenhum dado é criado para contornar a ausência de imagens.

## Treinamento

### Obter os pesos de partida

Escolha um checkpoint de **detecção de objetos** compatível com Ultralytics. Um ponto de partida possível é `yolo11n.pt`, documentado como modelo de detecção na [página oficial do YOLO11](https://docs.ultralytics.com/models/yolo11/). Obtenha esse arquivo explicitamente a partir da fonte oficial e guarde-o, por exemplo, em `modelos/base/yolo11n.pt`.

Os pesos base servem para fine-tuning; eles não são apresentados como detector de plástico do projeto. O script exige um arquivo `.pt` local existente e não usa o nome do modelo como autorização para baixá-lo. Use checkpoints de origem confiável, pois arquivos PyTorch contêm objetos serializados.

### Executar fine-tuning

Após preencher e revisar `train` e `val`:

```bash
python -m modelo.treinamento --dataset dataset/plastico.yaml --pesos-base modelos/base/yolo11n.pt --epochs 50 --imgsz 640 --batch 8 --device cpu --seed 42
```

Os números do comando são parâmetros de execução de exemplo, não resultados ou garantia de desempenho. Escolha o número de épocas e a resolução com base em experimentos sobre o conjunto de validação. Se faltar memória, reduza `--batch` ou `--imgsz`. `imgsz` deve ser um múltiplo de 32, pelo menos 32; batch e epochs devem ser inteiros positivos.

Para uma GPU suportada e uma instalação compatível do PyTorch, use `--device 0`. A preparação de drivers/CUDA depende do equipamento; a interface mantém CPU como opção padrão.

O script realiza validação antes de carregar o modelo, usa `workers=0` para facilitar a execução no Windows e desativa AMP (`amp=False`) para evitar o carregamento do modelo auxiliar usado na verificação AMP. Também utiliza `plots=False`, `cache=False`, seed explícita e opção determinística. Esses parâmetros não garantem reprodução bit a bit entre hardwares ou versões diferentes.

O treinamento efetivo usa a [API de treinamento do Ultralytics](https://docs.ultralytics.com/modes/train/). Ao concluir com sucesso:

- O YOLO mantém os artefatos da execução em `resultados/treinamentos/treino_<identificador>/yolo/`.
- O melhor checkpoint `best.pt` é verificado e copiado para `modelos/plastico_treino_<identificador>.pt`.
- Um `treinamento.json` registra parâmetros, versões, hashes, manifesto dos dados e caminhos dos artefatos.
- O terminal mostra o caminho do melhor modelo copiado. O nome exclusivo evita substituir modelos anteriores.

Se o dataset estiver vazio, incompleto ou inválido, o script termina com código de saída 1 e uma explicação. Não há treinamento de demonstração nem promoção de um modelo substituto.

Embora os pesos sejam locais e o YAML não permita downloads de dados, algumas versões do Ultralytics podem tentar obter recursos auxiliares, como fontes, durante treino ou avaliação. Portanto, não se promete execução totalmente sem rede dessas rotinas apenas por usar pesos locais. Não há download silencioso de um modelo genérico para apresentar detecções na aplicação.

## Validação e avaliação

Use a partição de validação durante o desenvolvimento, substituindo `SEU_MODELO.pt` pelo nome real gerado no treinamento:

```bash
python -m modelo.avaliacao --dataset dataset/plastico.yaml --pesos modelos/SEU_MODELO.pt --split val --imgsz 640 --batch 8 --device cpu
```

Depois de congelar as decisões do experimento, avalie no conjunto de teste independente:

```bash
python -m modelo.avaliacao --dataset dataset/plastico.yaml --pesos modelos/SEU_MODELO.pt --split test --imgsz 640 --batch 8 --device cpu
```

O padrão é `--split test`. A avaliação exige imagens e anotações válidas no split escolhido, incluindo exemplos positivos. O modelo precisa ter as mesmas classes declaradas no YAML. Sem esses requisitos, nenhum JSON com métricas é produzido.

As métricas são extraídas da execução real de `model.val`, conforme a [API de validação do Ultralytics](https://docs.ultralytics.com/modes/val/):

| Campo salvo | Origem no retorno YOLO | Interpretação |
| --- | --- | --- |
| `precision` | `resultado.box.mp` | Resumo de precisão fornecido pelo avaliador |
| `recall` | `resultado.box.mr` | Resumo de recall fornecido pelo avaliador |
| `mAP50` | `resultado.box.map50` | AP média no limiar de IoU 0,50 |
| `mAP50-95` | `resultado.box.map` | AP média nos limiares de IoU de 0,50 a 0,95 |

Os valores são armazenados na escala de 0 a 1. Não são substituídos por zero quando ausentes ou inválidos. Zero só pode ser salvo quando retornado como métrica válida da avaliação executada.

O resultado fica em `resultados/metricas/avaliacao_<split>_<identificador>/metricas.json`, com parâmetros, classes, split, contagens, hashes, manifesto e versões. O argumento `--conf`, por padrão 0.001, é um limiar baixo para a avaliação; não precisa ser igual ao limiar escolhido para exibir caixas na interface. `--iou`, por padrão 0.7, é o parâmetro de pós-processamento e não substitui os limiares usados no cálculo de mAP.

Os resultados futuros devem ser interpretados para o conjunto e as condições registrados. A existência de métricas não demonstra generalização para qualquer praia, rio ou condição de captura.

## Utilizar um modelo já treinado

1. Obtenha pesos `.pt` de detecção treinados com a classe `0: plastico` e registre sua origem e avaliação.
2. Coloque o arquivo diretamente em `modelos/`, por exemplo `modelos/best.pt`. Arquivos dentro de `modelos/base/` não aparecem na lista da interface.
3. Envie uma fotografia e clique em **Atualizar lista de modelos**, ou recarregue a aplicação.
4. Selecione os pesos e clique em **Detectar resíduos**.

O aplicativo verifica arquivo existente, extensão, carregamento, tarefa e nomes de classes. Um modelo genérico COCO não é renomeado para parecer um detector de plástico. O nome da classe e a existência do arquivo também não provam que o treinamento foi adequado; essa evidência depende dos registros e da avaliação da equipe.

Use caminhos sem apóstrofos e nomes personalizados para os pesos, como `best.pt`. O carregador rejeita caminhos que o Ultralytics alteraria automaticamente, evitando procurar outro arquivo por um nome transformado.

## Salvar resultados

Após uma inferência real, clique em **Salvar imagens e informações em resultados/**. Cada gravação cria um identificador de data/hora UTC com sufixo único e preserva os resultados anteriores:

```text
resultados/
├── deteccoes/<identificador>/
│   ├── original.jpg ou original.png
│   ├── deteccoes.png
│   └── deteccoes.json
└── filtros/<mesmo_identificador>/
    ├── original_comparacao.png
    ├── gaussiano.png
    ├── mediana.png
    └── segmentacao.png
```

O JSON reúne caixas, classes, confianças, quantidade efetivamente retornada, hash da imagem de entrada, identificação e hash dos pesos, parâmetros, resolução e tempo da inferência. Os parâmetros dos filtros e a resolução da comparação também são registrados. O arquivo original é preservado em seus bytes enviados.

Os botões **Baixar imagem com detecções** e **Baixar informações JSON** oferecem exportação pelo navegador. A gravação em `resultados/` acontece na máquina que está executando o Streamlit; o download acontece no navegador do usuário. Resultados de uma inferência real sem objetos também podem ser salvos, com lista vazia e a devida interpretação.

## Testes

```bash
python -m pytest
```

Para executar apenas o processamento e os contratos do detector:

```bash
python -m pytest tests/test_preprocessamento.py tests/test_detector.py
```

Os testes não precisam de pesos YOLO treinados. As matrizes e os arquivos temporários usados para verificar leitura, filtros, anotações e contratos são **fixtures técnicas de teste**, não fotografias de treinamento ou evidência de desempenho. Os objetos que substituem o retorno do YOLO nos testes verificam a integração; seus valores não são métricas do detector.

A suíte contempla leitura válida e inválida, redimensionamento, cores, filtros, segmentação, ausência e incompatibilidade de modelo, persistência, validação de dados e comportamento da interface. Os testes de Streamlit usam `AppTest`, incluindo upload técnico em memória. Ambientes isolados que bloqueiem sockets locais podem precisar permitir essa comunicação para executar os testes da interface.

Rodar a suíte com sucesso verifica comportamento do software; não valida a qualidade científica das detecções. A avaliação do modelo permanece pendente até existirem dados e pesos reais.

O registro das verificações realizadas na entrega e das versões utilizadas está em [VALIDACAO.md](VALIDACAO.md).

## Tratamento de erros

| Situação | Ação recomendada |
| --- | --- |
| Python ou Streamlit não encontrado | Confirme a instalação do Python e use o executável da `.venv` com `-m streamlit`. |
| Biblioteca ausente | Execute `python -m pip install -r requirements.txt` no mesmo ambiente que executa o app. |
| Arquivo inválido ou corrompido | Abra a fotografia em um visualizador, exporte JPG/PNG válido e tente novamente. |
| Imagem acima dos limites | Reduza tamanho/resolução antes do upload, preservando o arquivo original da coleta. |
| Modelo inexistente | Continue usando processamento; adicione os pesos personalizados quando disponíveis. |
| Falha de carregamento ou classes incompatíveis | Verifique origem, integridade, versão do Ultralytics e o mapeamento `0: plastico`. |
| Dataset vazio ou label ausente | Inclua fotografias reais e revise as anotações correspondentes; não crie labels vazios para ocultar falta de anotação. |
| YAML inválido | Confira indentação, campos permitidos, classes e caminhos locais. |
| Nenhuma detecção | Examine a imagem, o domínio de treinamento e os parâmetros; a contagem zero não prova ausência de resíduos. |
| Memória insuficiente no treino/avaliação | Reduza batch/imgsz e confira o dispositivo selecionado. |
| Falha ao salvar | Verifique espaço e permissão de escrita na pasta do projeto. |

Ajuda dos scripts:

```bash
python -m modelo.treinamento --help
python -m modelo.avaliacao --help
python -m pip check
```

## Git e colaboração

O `.gitignore` exclui ambientes virtuais, caches Python, arquivos `.env`, pesos, datasets pesados e resultados gerados. Mantém a configuração YAML, o código, os testes e os marcadores das pastas vazias. Dados e modelos devem ter armazenamento e controle de versão próprios, com referência de origem e hash quando usados em experimentos.

Combine mudanças nos contratos dos módulos antes de alterar seus consumidores. Para comparar experimentos, preserve o dataset, o split, os parâmetros, as versões e os checkpoints utilizados. Não sobrescreva a partição de teste para melhorar artificialmente a avaliação.

## Limitações atuais e próximos passos

- **Dados e detector pendentes:** não foram fornecidas fotografias anotadas nem pesos personalizados; não há desempenho de detecção medido nesta entrega.
- **Aparência não confirma material:** a classificação se baseia em pistas visuais e pode confundir vidro, metal, tecido, vegetação, espuma e outros materiais.
- **Condições de captura:** objetos pequenos, parcialmente cobertos, submersos, desfocados ou distantes podem não ser detectados.
- **Segmentação tradicional:** separa intensidades, podendo destacar reflexos, sombras e areia; não identifica plástico.
- **Uma classe no MVP:** todas as caixas de resíduos plásticos anotados usam `plastico`. A aplicação não distingue tipos de plástico nem polímeros.
- **Sem microplásticos:** o sistema não realiza análise microscópica, laboratorial ou química.
- **Sem comparação experimental de filtros:** a interface permite observação; qualquer conclusão de superioridade exige protocolo e medições reais.

Próximos passos da equipe: definir o protocolo de anotação, coletar e revisar dados autorizados, separar partições, treinar o modelo, examinar erros na validação, registrar a avaliação final no teste e documentar limitações observadas. Estudos futuros podem comparar o efeito dos filtros sobre a detecção sob condições controladas, mantendo o mesmo protocolo de avaliação.

Para expandir para garrafas, sacolas, copos/recipientes, embalagens e outros resíduos, será necessário definir classes mutuamente claras, revisar/reanotar o dataset, atualizar `nc` e `names`, treinar novos pesos e adaptar o mapeamento esperado pela interface. `DetectorPlastico` aceita um mapeamento de classes explícito para essa extensão; a interface entregue mantém o contrato de uma classe. A expansão não é necessária para executar o MVP.
