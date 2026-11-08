# Anotações recebidas

`annotations.json` foi copiado, byte a byte, de `APS/TACO/data/annotations.json` do APS.zip fornecido em 08/11/2026.

SHA-256: `812c2e3f0d336b437f276597715de2abb08bb8a641599c45f593d233ec66457b`.

O JSON original descreve 1.500 imagens e 4.784 anotações de resíduos diversos. A política explícita de plástico do preparador seleciona 24 categorias e 2.245 caixas desse arquivo. Quatro dessas caixas precisam de interseção com os limites da imagem. Esses números foram calculados sobre os dados reais recebidos; não são métricas de desempenho.

As fotos não foram duplicadas no Git. O material recebido contém 1.050 imagens de treino, 300 de validação e 150 de teste em `dataset_plasticos.zip`, dentro de APS.zip. O notebook usa esse ZIP pelo Drive; a linha de comando também aceita diretamente APS.zip.

Fonte original: [TACO](https://github.com/pedropro/TACO), Pedro F. Proença e Pedro Simões, [artigo](https://arxiv.org/abs/2003.06975). As informações de licenciamento e origem das imagens são mantidas no JSON. Não se presume que a licença do código do toolkit substitua as licenças das imagens.

Antes do treinamento, revise o escopo das categorias, as caixas e os negativos. A preparação não constitui treinamento ou validação científica do detector.
