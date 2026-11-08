# Detector experimental incluído

`plastico_yolo11n_taco.pt` é uma exportação do melhor checkpoint segundo a validação, obtido em 8 de novembro de 2026. A arquitetura é YOLO11n e a classe única é `0: plastico`. O checkpoint foi escolhido na época 24 de uma execução interrompida a pedido do usuário após 25 épocas completas, das 50 originalmente previstas. Não houve novo treinamento durante a integração.

O fine-tuning utilizou 1.050 fotos de treino e 300 de validação do material TACO enviado em `APS.zip`, com as caixas COCO convertidas conforme `treinamento/preparacao.py`. O conjunto reservado de teste contém 150 fotos, com 221 objetos plásticos anotados. As categorias incluídas abrangem espuma/isopor e embalagens multicamadas; não existe confirmação química de material. A seleção de categorias precisa de revisão visual humana. A origem das anotações está em `dataset/taco/PROVENIENCIA.md`.

## Avaliação real no teste

| Métrica | Resultado |
| --- | --- |
| Precisão agregada | 48,50% |
| Recall agregado | 41,63% |
| mAP com IoU 0,50 | 35,64% |
| mAP com IoU de 0,50 a 0,95 | 24,33% |

Parâmetros: resolução 640, batch 4, confiança 0,001 para construir as curvas de avaliação e IoU de NMS 0,7. A interface usa confiança 0,25 por padrão, portanto os agregados de precisão/recall do avaliador não representam exatamente esse limiar da interface. Não representam acurácia global nem probabilidade de acerto de uma caixa. Valores completos, versões e hashes constam em `avaliacao_teste.json`.

O Ultralytics restaurou automaticamente sete JPEGs sem marcador final na cópia de trabalho usada para avaliação. Os hashes anteriores e posteriores estão no mesmo JSON. O `APS.zip` original permanece intacto.

A divisão recebida é aleatória e pode conter cenas semelhantes entre partições. Estes resultados descrevem o conjunto TACO reservado e não comprovam generalização para outras praias e rios. O modelo pode confundir outros materiais e omitir plásticos pequenos, distantes, parcialmente cobertos ou desfocados. Ele não detecta microplásticos. Não foi escolhido nem ajustado a partir do resultado do teste.

## Exportação e reprodução

O artefato de inferência foi gerado com `ultralytics.utils.torch_utils.strip_optimizer`, em arquivo separado: mantém o modelo EMA em FP16 e remove o estado do otimizador, reduzindo aproximadamente 26,6 MB para 5,5 MB. A inferência no aplicativo usa CPU. Os checkpoints originais `best.pt` e `last.pt`, inclusive o otimizador para possível retomada, permanecem preservados fora do Git.

`modelo.json` identifica a época, o hash do checkpoint original, o hash do arquivo exportado e as versões do treino. A implementação Ultralytics registra AGPL-3.0 nos metadados dos pesos; consulte os termos do projeto Ultralytics ao redistribuir o modelo. As licenças das fotografias e anotações são documentadas separadamente na proveniência TACO.

Para testar pelo aplicativo, envie uma foto, mantenha os parâmetros iniciais e clique em **Detectar resíduos**. Confira as caixas visualmente e use os downloads da imagem e do JSON para registrar os resultados. Nenhuma detecção não comprova ausência de plástico.
