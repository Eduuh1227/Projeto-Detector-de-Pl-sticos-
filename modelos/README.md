# Modelos disponíveis

O seletor da interface oferece **Reforçado — experimental** por padrão e **Anterior — TACO** para comparação. Ambos exigem revisão visual dos resultados, usam uma única classe e executam inferência em CPU no site.

## Reforçado — experimental, disponibilizado em 09/11/2026

`plastico_yolo11n_reforco.pt` é a cópia de inferência de `last.pt` da terceira rodada, época 15. Foi escolhido para o teste solicitado pelo usuário, pois a checagem final marcou o pote da imagem 1 e as garrafas das imagens 2 e 3. Não é anunciado como o melhor modelo geral. O modelo completou as 50 épocas iniciais e mais três rodadas de 15, totalizando 95 épocas. As três imagens ficaram fora do treinamento, mas influenciaram as checagens de desenvolvimento e não são teste independente.

O conjunto de reforço contém 1.771 imagens de treino e 368 de validação: fotos TACO originais e recortes de fotografias reais com anotações existentes. As 150 imagens TACO de teste foram preservadas. Fonte externa: [Plastic Bottles Dataset](https://github.com/m0-n/Plastic-Bottles-Dataset/tree/76b84abc4c8180f7ee0a67abfee917185d925776), cujos dados são declarados em domínio público pelo autor. Os recortes são versões das mesmas cenas e não aumentam na mesma proporção o número de observações independentes. O protocolo e suas limitações constam em `treinamento/REFORCO.md`.

Em 300 fotos TACO de validação e 515 objetos anotados, com confiança 0,25, resolução 640 e correspondência um-a-um de IoU mínimo 0,50: 189 acertos, 194 marcações sem correspondência e 326 omissões. Precisão 49,35%, recall 36,70%, F1 42,09%. O F1 da referência de 50 épocas é 43,03%; portanto, o reforçado **não passou no critério automático de qualidade geral**. A publicação experimental foi solicitada pelo usuário mesmo com essa limitação. Marcações sem referência podem incluir objetos não anotados, e as referências manuais das três imagens são aproximadas.

`reforco.json` identifica os pesos e a origem. `avaliacao_reforco.json` registra os valores completos e as checagens. Não há mAP de teste final para essa versão. A exportação mantém o modelo EMA e remove o otimizador, sem alterar os checkpoints originais preservados fora do Git. Os metadados de treinamento completos ficam na cópia local.

## Anterior — TACO, publicado após 25 épocas

`plastico_yolo11n_taco.pt` é uma exportação do melhor checkpoint segundo a validação, obtido em 8 de novembro de 2026. A arquitetura é YOLO11n e a classe única é `0: plastico`. O checkpoint foi escolhido na época 24 de uma execução interrompida a pedido do usuário após 25 épocas completas, das 50 originalmente previstas. Não houve novo treinamento durante a integração.

O fine-tuning utilizou 1.050 fotos de treino e 300 de validação do material TACO enviado em `APS.zip`, com as caixas COCO convertidas conforme `treinamento/preparacao.py`. O conjunto reservado de teste contém 150 fotos, com 221 objetos plásticos anotados. As categorias incluídas abrangem espuma/isopor e embalagens multicamadas; não existe confirmação química de material. A seleção de categorias precisa de revisão visual humana. A origem das anotações está em `dataset/taco/PROVENIENCIA.md`.

### Avaliação real no teste — somente modelo anterior

| Métrica | Resultado |
| --- | --- |
| Precisão agregada | 48,50% |
| Recall agregado | 41,63% |
| mAP com IoU 0,50 | 35,64% |
| mAP com IoU de 0,50 a 0,95 | 24,33% |

Parâmetros: resolução 640, batch 4, confiança 0,001 para construir as curvas de avaliação e IoU de NMS 0,7. A interface usa confiança 0,25 por padrão, portanto os agregados de precisão/recall do avaliador não representam exatamente esse limiar da interface. Não representam acurácia global nem probabilidade de acerto de uma caixa. Valores completos, versões e hashes constam em `avaliacao_teste.json`.

O Ultralytics restaurou automaticamente sete JPEGs sem marcador final na cópia de trabalho usada para avaliação. Os hashes anteriores e posteriores estão no mesmo JSON. O `APS.zip` original permanece intacto.

A divisão recebida é aleatória e pode conter cenas semelhantes entre partições. Estes resultados descrevem o conjunto TACO reservado e não comprovam generalização para outras praias e rios. O modelo pode confundir outros materiais e omitir plásticos pequenos, distantes, parcialmente cobertos ou desfocados. Ele não detecta microplásticos. Não foi escolhido nem ajustado a partir do resultado do teste.

### Exportação e reprodução — modelo anterior

O artefato de inferência foi gerado com `ultralytics.utils.torch_utils.strip_optimizer`, em arquivo separado: mantém o modelo EMA em FP16 e remove o estado do otimizador, reduzindo aproximadamente 26,6 MB para 5,5 MB. A inferência no aplicativo usa CPU. Os checkpoints originais `best.pt` e `last.pt`, inclusive o otimizador para possível retomada, permanecem preservados fora do Git.

`modelo.json` identifica a época, o hash do checkpoint original, o hash do arquivo exportado e as versões do treino. A implementação Ultralytics registra AGPL-3.0 nos metadados dos pesos; consulte os termos do projeto Ultralytics ao redistribuir o modelo. As licenças das fotografias e anotações são documentadas separadamente na proveniência TACO.

Para testar pelo aplicativo, envie uma foto, mantenha os parâmetros iniciais e clique em **Detectar resíduos**. Confira as caixas visualmente e use os downloads da imagem e do JSON para registrar os resultados. Nenhuma detecção não comprova ausência de plástico.
