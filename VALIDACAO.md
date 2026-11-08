# Registro de validação do software

## Integração do modelo real — 08/11/2026

O treino foi interrompido a pedido do usuário após 25 épocas completas. O checkpoint da época 24 foi selecionado pela validação, exportado separadamente sem otimizador e incluído em `modelos/plastico_yolo11n_taco.pt`. Os checkpoints de treino permanecem preservados. Os hashes e a origem constam em `modelos/modelo.json`.

A avaliação real nas 150 fotos reservadas para teste, com 221 objetos anotados, retornou precisão 0,4850039821, recall 0,4162895928, mAP50 0,3564273370 e mAP50–95 0,2433127917. Os parâmetros e as limitações estão em `modelos/avaliacao_teste.json`. Durante essa avaliação, o Ultralytics restaurou sete JPEGs sem marcador final na cópia de trabalho; foram registrados os hashes anteriores e posteriores. O arquivo original `APS.zip` não foi alterado.

Na revisão da integração, a suíte completa passou com **129 testes aprovados em 6,79 segundos**. Um teste adicional de utilização executou a interface Streamlit com o detector real em CPU, usando uma cópia reduzida a 1280 pixels de uma fotografia real de validação (`taco_102.jpg`). O botão Detectar resíduos produziu três caixas e os downloads de imagem e JSON. Essa verificação funcional não constitui medida de qualidade do detector. O modelo continua experimental, com erros e omissões.

As versões de inferência foram alinhadas a Ultralytics 8.4.191 e PyTorch 2.14.1. Os registros de 27/10 abaixo descrevem a entrega anterior, quando não havia pesos.

## Preparação do material recebido — 08/11/2026

As anotações TACO recebidas foram incorporadas sem alteração de bytes, com SHA-256 registrado em `dataset/taco/PROVENIENCIA.md`. Foram conferidas matematicamente as 2.245 caixas das 24 categorias selecionadas; quatro caixas precisaram de interseção com os limites da imagem. Passaram 5 testes técnicos do novo preparador, incluindo seleção de categorias, geometria e extração seletiva de ZIP. As 10 células de código do notebook foram compiladas para verificar sintaxe; não foram executadas em Colab/GPU e não possuem saídas de treinamento. A decodificação completa das 1.500 fotos e o treinamento permanecem para a execução do roteiro. As verificações anteriores estão registradas abaixo como histórico.

Data: **27/10/2026**. Ambiente: Windows, Python **3.12.14**, CPU.

Este registro descreve verificações do código. Não é uma avaliação científica do detector de plástico.

## Verificações executadas

| Verificação | Resultado observado |
| --- | --- |
| Compilação dos módulos Python | Sem erros de sintaxe |
| `python -m pip check` | Nenhum conflito de dependências encontrado |
| Suíte completa `python -m pytest` | **124 testes aprovados**, sem falhas, em 16,01 segundos na revisão para hospedagem |
| Importação real de PyTorch e Ultralytics | Concluída; PyTorch instalado para CPU |
| Aplicação Streamlit em navegador local | Página inicial exibida com upload, aviso de modelo ausente e informações acadêmicas |
| Treinamento com dataset vazio | Código de saída 1 e mensagem orientando adicionar/anotar fotografias; nenhum treinamento executado |
| Avaliação com teste vazio | Código de saída 1 e mensagem orientando preencher o conjunto de teste; nenhuma métrica criada |
| Inspeção das pastas entregues | Nenhuma imagem de dataset, label, peso ou resultado científico incluído |

Os testes incluem leitura JPG/PNG, arquivos inválidos e corrompidos, caminhos com acentos, cores, transparência, proporção, filtros, limiarização, morfologia, parâmetros inválidos, ausência de modelo, classes incompatíveis, contratos de inferência, salvamento sem sobrescrita, configuração YAML, labels, duplicatas entre partições e métricas inválidas.

Os testes de utilização executam o Streamlit com `AppTest`: abertura sem pesos, upload técnico válido e inválido, controles de processamento/histograma, habilitação do botão ao encontrar pesos, apresentação de respostas de um substituto de teste, invalidação do resultado ao alterar confiança e tratamento de erros. O navegador foi utilizado para conferir a página inicial real; as interações de upload foram verificadas no `AppTest`.

A revisão para hospedagem acrescenta verificação do ZIP em memória, referências dos arquivos exportados, ausência de gravação de resultados no modo hospedado, bloqueio de downloads acima do limite e rejeição de upload maior que o limite configurado. `vercel.json` foi lido como JSON válido, e `git diff --check` não apontou erros. O build do container e o fluxo pela URL pública precisam de validação após a implantação; estes testes locais não comprovam a conclusão do deploy.

## Limite da validação

**Na entrega de 27/10 não foram executados treinamento, avaliação ou detecção real de plástico.** Naquele momento não havia dataset anotado nem checkpoint personalizado fornecido. A revisão de 08/11 no início deste documento registra o treinamento parcial, a avaliação e a inferência reais posteriores.

As matrizes de pixels, labels e arquivos marcadores dos testes são fixtures técnicas temporárias. Os substitutos do YOLO verificam apenas os contratos entre módulos. Seus valores não representam objetos encontrados em fotografias nem desempenho do modelo. Esses arquivos temporários ficam fora das pastas entregues de dataset/modelos/resultados e não integram o ZIP.

O fluxo com pesos reais deverá ser validado novamente após a coleta/anotação, o treinamento e a avaliação em imagens independentes. A ausência de conflito de bibliotecas e a aprovação dos testes não substituem essa etapa.

## Versões utilizadas

| Pacote | Versão observada |
| --- | --- |
| NumPy | 2.5.4 |
| OpenCV (`opencv-python`) | 4.14.0.94 |
| Matplotlib | 3.11.2 |
| Ultralytics | 8.4.190 |
| Streamlit | 1.66.1 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| PyTorch | 2.14.1+cpu |

`requirements.txt` usa intervalos de versões. Uma instalação futura pode resolver versões diferentes; registre-as com `python -m pip freeze` e execute novamente `python -m pytest`. Não é necessário instalar GPU/CUDA para testar o processamento ou executar a interface.

Em ambientes restritos, os testes Streamlit precisam de comunicação por sockets locais para inicializar o runtime. A execução final foi realizada com essa comunicação permitida; isso não é uma exigência de conexão com um serviço externo.
