# Preparação e treinamento com os arquivos recebidos

O arquivo APS.zip enviado contém o projeto, as fotos originais do TACO, uma cópia das fotos no formato de pastas YOLO e dataset_plasticos.zip. Seu tamanho é 14.341.052.163 bytes; o ZIP interno tem 4.737.450.897 bytes. Basta usar o ZIP interno no Colab. O comando `files.upload()` da captura apenas envia arquivos; não treina o modelo. Para esses volumes, prefira Google Drive.

Foram observadas 1.050 fotos em train, 300 em val e 150 em test. Essas são contagens dos arquivos recebidos, não resultados de desempenho. Não existe checkpoint treinado no material analisado.

## Arquivos incorporados

- `dataset/taco/annotations.json`: anotações COCO originais, copiadas do material enviado, sem modificar seu conteúdo.
- `preparar_dataset_taco.py`: entrada de linha de comando para APS.zip ou dataset_plasticos.zip.
- `treinamento/preparacao.py`: extração seletiva das fotos e conversão revisada.
- `notebooks/Treinar_YOLO_Colab.ipynb`: roteiro executável, sem resultados ou pesos simulados.

O áudio e as capturas são referências de orientação, não dependências do treinamento. As fotos e os ZIPs grandes permanecem externos ao Git; o preparador os lê diretamente do arquivo fornecido. O site também não faz treinamento por upload de visitantes.

## Revisão da conversão recebida

O script recebido selecionava IDs fixos e descartava categorias com menos de 20 anotações. O ID 28 corresponde a Metal lid, enquanto Plastic lid é 27. O limite de frequência acabou impedindo a inclusão de metal neste arquivo, mas também omitiu plásticos, incluindo tampas e categorias raras. Quatro caixas selecionadas tinham coordenadas fora da imagem.

A versão incorporada usa nomes e supercategorias do JSON, sem descartar categorias raras. Intersecciona cada caixa com os limites da imagem antes de normalizar e registra as alterações em `conversao_taco.json`. Preserva a divisão de imagens recebida e não sobrescreve um dataset já preparado. Não cria fotografias ou anotações sem origem.

**Revise a política de material:** a seleção inclui embalagens e espuma plástica/isopor. Categorias ambíguas como Unlabeled litter e Rope & strings não são resolvidas automaticamente. Um label vazio significa ausência de classes selecionadas nas anotações COCO, não comprovação visual de ausência de plástico. Revise negativos e casos ambíguos antes de usar os dados em um experimento final. A divisão recebida é aleatória; verifique cenas próximas/locais compartilhados entre partições, além da checagem de duplicatas exatas do validador.

## No Google Colab

1. Extraia apenas `dataset_plasticos.zip` de APS.zip e envie-o ao seu Google Drive, em uma pasta `APS`.
2. Abra o notebook deste repositório no Colab e selecione GPU em Ambiente de execução → Alterar tipo de ambiente de execução.
3. Execute as células de instalação, montagem do Drive e configuração do caminho do ZIP.
4. Prepare e valide os dados. Confira a política de categorias, revise as anotações e marque a confirmação de revisão na célula de treinamento.
5. Execute o treinamento. O notebook salva execuções e pesos no Drive.
6. Use val durante o desenvolvimento. Execute a avaliação test após fixar suas decisões; não ajuste parâmetros pelos resultados do teste.
7. Baixe o checkpoint produzido. Incluí-lo no site exige uma nova implantação com o arquivo de pesos.

GPU e duração da sessão dependem da disponibilidade do Colab. Não há garantia de duração, acurácia ou desempenho. O notebook não foi executado em GPU nesta entrega.

## No computador

Na raiz do repositório, com o ambiente e as dependências instalados:

```powershell
python preparar_dataset_taco.py --arquivo "C:\caminho\APS.zip" --saida dados_treinamento
python -c "from modelo.dados import validar_dataset; print(validar_dataset('dados_treinamento/plastico.yaml', ('train','val','test')).contagens)"
python -m modelo.treinamento --dataset dados_treinamento/plastico.yaml --pesos-base modelos/base/yolo11n.pt --epochs 50 --imgsz 640 --batch 8 --device cpu
```

Obtenha o checkpoint base yolo11n.pt explicitamente da fonte oficial antes desse comando. Substitua cpu por 0 somente em um ambiente PyTorch com GPU compatível. Após o treinamento, utilize o caminho real impresso:

```powershell
python -m modelo.avaliacao --dataset dados_treinamento/plastico.yaml --pesos modelos/NOME_REAL.pt --split test
```

## Proveniência

Fonte: [TACO — Trash Annotations in Context](https://github.com/pedropro/TACO), de Pedro F. Proença e Pedro Simões; artigo: https://arxiv.org/abs/2003.06975. Preserve os créditos e confira as licenças e metadados das imagens no JSON original. O notebook utiliza o código deste repositório e pesos base Ultralytics explicitamente identificados. Não foi gerado treinamento, precision, recall ou mAP nesta preparação.
