# Protocolo do treinamento reforçado

O usuário solicitou mais treinamento devido a omissões em três imagens de desenvolvimento. A execução adicional foi feita entre 08 e 09/11/2026, a partir do melhor modelo da continuação até 50 épocas. As três imagens do usuário não entraram no treino, mas foram checadas repetidamente para decidir se o objetivo específico havia sido atingido; não constituem teste científico independente.

## Dados

- TACO recebido em APS.zip: mantidas 1.050 imagens de treino, 300 de validação e 150 de teste.
- Fonte adicional: [m0-n/Plastic-Bottles-Dataset](https://github.com/m0-n/Plastic-Bottles-Dataset/tree/76b84abc4c8180f7ee0a67abfee917185d925776), commit congelado `76b84abc4c8180f7ee0a67abfee917185d925776`, com anotações XML de garrafas plásticas. O autor declara os dados em domínio público. Nenhum código dessa fonte foi executado.
- Foram gerados 398 recortes reais para treino e 68 para validação externa. Os grupos da Índia 003, 016 e 023 ficaram na validação externa; os demais grupos foram usados para treino. Foram excluídos arquivos inconsistentes e duplicatas identificadas.
- Mais 323 recortes de garrafas e recipientes das imagens TACO de treino reforçaram a representação desses objetos. Não foram usados recortes de TACO val/test para treinar.
- Total: 1.771 arquivos de treino com 2.579 caixas, 368 de validação com 590 caixas e 150 de teste com 221 caixas. Recortes da mesma cena não são observações independentes.

Cada recorte utiliza pixels de uma foto real e suas caixas já anotadas. São preservadas caixas com pelo menos 50% da área dentro do recorte; pode ocorrer redução proporcional para 1280 pixels no maior lado e codificação JPEG 95, sem ampliar a resolução. A fonte externa anota garrafas; outros resíduos podem existir sem caixa. Esse problema foi reduzido pelos recortes, mas não eliminado. A revisão visual realizada cobriu amostras, não todo o conjunto. A divisão TACO recebida permaneceu com suas limitações de cenas semelhantes.

## Rodadas

YOLO11n, uma classe `plastico`, resolução 640, batch 4, GPU RTX 5060, AMP desativado, zero workers, sem cache de imagens, AdamW, seed 42, `scale=0.25`, `translate=0.05`, `degrees=10`, `lrf=0.1`, `warmup_epochs=1`, `close_mosaic=5`.

| Rodada | Épocas | Camadas iniciais congeladas | Taxa inicial | Mosaic |
| --- | --- | --- | --- | --- |
| 1 | 15 | 10 | 0,0005 | 0,15 |
| 2 | 15 | 0 | 0,0002 | 0,10 |
| 3 | 15 | 0 | 0,0001 | 0 |

Cada rodada seguinte partiu do melhor checkpoint da validação combinada da rodada anterior. Os 45 ciclos adicionais terminaram. O checkpoint disponibilizado para testes é o **último da rodada 3**, pois foi esse o verificado nas três imagens finais; ele não foi escolhido por superar o melhor F1 geral.

## Critérios e resultado

A cada cinco épocas, as três imagens foram inferidas em CPU com confiança 0,25 e resolução 640. Usaram-se referências visuais aproximadas para o pote da imagem 1 e as garrafas inteiras das imagens 2 e 3, com IoU mínimo 0,50. Marcar só uma tampa ou um reflexo não atendia ao critério. Essas referências não são labels de treinamento nem foram apresentadas como anotações fornecidas pelo usuário.

Quando os três alvos eram detectados, comparava-se o modelo nas 300 fotos TACO de validação: pelo menos 179 objetos encontrados e F1 igual ou superior a `2×179/(2×179+138+336)`, resultado da referência de 50 épocas. O checkpoint final encontrou os três alvos, mas teve 189 acertos, 194 marcações sem correspondência, 326 omissões e F1 0,4209354, abaixo de 0,4302885. Não passou no critério automático; foi disponibilizado como **experimental a pedido do usuário**.

O modelo anterior preservado no seletor é o publicado após 25 épocas, diferente da referência de 50 épocas usada no critério. As métricas de teste do modelo anterior não devem ser atribuídas ao reforçado. Este protocolo não comprova detecção de todos os plásticos, de material químico ou de microplásticos.

Os manifests detalhados dos recortes, fontes originais, checkpoints, scripts locais da execução e checagens ficam na pasta de trabalho `outputs/treino-reforco`. As fotos, os ZIPs e os checkpoints de treino não integram o Git; estão incluídos apenas os arquivos de inferência e relatórios resumidos necessários para testar o site.
