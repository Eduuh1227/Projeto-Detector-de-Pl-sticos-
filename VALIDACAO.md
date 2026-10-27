# Registro de validação do software

Data: **27/10/2026**. Ambiente: Windows, Python **3.12.14**, CPU.

Este registro descreve verificações do código. Não é uma avaliação científica do detector de plástico.

## Verificações executadas

| Verificação | Resultado observado |
| --- | --- |
| Compilação dos módulos Python | Sem erros de sintaxe |
| `python -m pip check` | Nenhum conflito de dependências encontrado |
| Suíte completa `python -m pytest` | **117 testes aprovados**, sem falhas, em 6,44 segundos nesta execução |
| Importação real de PyTorch e Ultralytics | Concluída; PyTorch instalado para CPU |
| Aplicação Streamlit em navegador local | Página inicial exibida com upload, aviso de modelo ausente e informações acadêmicas |
| Treinamento com dataset vazio | Código de saída 1 e mensagem orientando adicionar/anotar fotografias; nenhum treinamento executado |
| Avaliação com teste vazio | Código de saída 1 e mensagem orientando preencher o conjunto de teste; nenhuma métrica criada |
| Inspeção das pastas entregues | Nenhuma imagem de dataset, label, peso ou resultado científico incluído |

Os testes incluem leitura JPG/PNG, arquivos inválidos e corrompidos, caminhos com acentos, cores, transparência, proporção, filtros, limiarização, morfologia, parâmetros inválidos, ausência de modelo, classes incompatíveis, contratos de inferência, salvamento sem sobrescrita, configuração YAML, labels, duplicatas entre partições e métricas inválidas.

Os testes de utilização executam o Streamlit com `AppTest`: abertura sem pesos, upload técnico válido e inválido, controles de processamento/histograma, habilitação do botão ao encontrar pesos, apresentação de respostas de um substituto de teste, invalidação do resultado ao alterar confiança e tratamento de erros. O navegador foi utilizado para conferir a página inicial real; as interações de upload foram verificadas no `AppTest`.

## Limite da validação

**Não foram executados treinamento, avaliação ou detecção real de plástico.** Nenhum dataset anotado nem checkpoint personalizado foi fornecido. Não há precision, recall, mAP ou acurácia medidos para este projeto.

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
