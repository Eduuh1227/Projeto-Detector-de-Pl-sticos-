"""Prepara uma cópia YOLO a partir das fotos e anotações reais fornecidas.

Não baixa, cria ou altera fotografias. Não executa código recebido no ZIP.
Não mede desempenho nem treina modelos. A seleção abaixo é uma política
explícita para revisão humana: inclui embalagens e espuma plástica (isopor).
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from zipfile import ZipFile

SPLITS = ("train", "val", "test")
EXTENSOES = {".jpg", ".jpeg", ".png"}
SUPERCATEGORIAS_PLASTICO = {
    "Plastic bag & wrapper", "Plastic container", "Other plastic",
    "Plastic glooves", "Plastic utensils", "Styrofoam piece",
}
NOMES_PLASTICO = {
    "Other plastic bottle", "Clear plastic bottle", "Plastic bottle cap",
    "Disposable plastic cup", "Other plastic cup", "Plastic lid",
    "Plastic straw", "Foam cup",
}


def ler_anotacoes(caminho: str | Path) -> dict:
    """Carrega o COCO original do TACO; falha se faltarem imagens/categorias/caixas."""
    coco = json.loads(Path(caminho).read_text(encoding="utf-8-sig"))
    if not isinstance(coco, dict) or any(c not in coco for c in ("images", "annotations", "categories")):
        raise ValueError("Use o TACO/data/annotations.json original presente em APS.zip.")
    return coco


def categorias_selecionadas(coco: dict) -> dict[int, str]:
    """Mantém todas as categorias da política, inclusive as pouco frequentes."""
    return {c["id"]: c["name"] for c in coco["categories"]
            if c["name"] in NOMES_PLASTICO or c.get("supercategory") in SUPERCATEGORIAS_PLASTICO}


def extrair_fotos(zip_dataset: str | Path, destino: str | Path) -> Path:
    """Extrai apenas fotos de images/train,val,test; ignora labels antigos e demais arquivos."""
    destino = Path(destino).resolve()
    if destino.exists():
        raise FileExistsError(f"A pasta de saída já existe: {destino}. Use outra pasta para preservar a cópia anterior.")
    itens = []
    chaves = set()
    with ZipFile(zip_dataset) as pacote:
        for entrada in pacote.infolist():
            nome = entrada.filename.replace("\\", "/")
            caminho = PurePosixPath(nome)
            if caminho.is_absolute() or ".." in caminho.parts or any(":" in parte for parte in caminho.parts):
                raise ValueError(f"Caminho inseguro no ZIP: {nome}")
            if stat.S_ISLNK(entrada.external_attr >> 16):
                raise ValueError("Links simbólicos não são aceitos no dataset.")
            partes = caminho.parts
            if entrada.is_dir() or caminho.suffix.lower() not in EXTENSOES:
                continue
            encontrados = [i for i, p in enumerate(partes)
                           if p == "images" and i + 2 < len(partes) and partes[i + 1] in SPLITS]
            if len(encontrados) != 1:
                continue
            relativo = Path(*partes[encontrados[0]:])
            chave = relativo.as_posix().casefold()
            if chave in chaves:
                raise ValueError(f"Duas entradas produzem o mesmo caminho: {relativo}")
            chaves.add(chave)
            itens.append((entrada, relativo))
        if not itens:
            raise ValueError("Não encontrei imagens em images/train, images/val ou images/test no ZIP.")
        necessario = sum(entrada.file_size for entrada, _ in itens)
        destino.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destino.parent).free < necessario + 1024**3:
            raise OSError("Espaço insuficiente para extrair as fotos com margem de 1 GB.")
        destino.mkdir()
        for indice, (entrada, relativo) in enumerate(itens, 1):
            arquivo = destino / relativo
            arquivo.parent.mkdir(parents=True, exist_ok=True)
            with pacote.open(entrada) as origem, arquivo.open("xb") as saida:
                shutil.copyfileobj(origem, saida, length=1024 * 1024)
            if indice % 100 == 0 or indice == len(itens):
                print(f"Fotos extraídas: {indice}/{len(itens)}", flush=True)
    return destino


def converter_caixa(bbox: list, largura_imagem: int, altura_imagem: int) -> tuple[list[float], bool]:
    """Intersecciona a caixa COCO com a imagem ANTES de normalizar centro/tamanho."""
    if len(bbox) != 4 or largura_imagem <= 0 or altura_imagem <= 0:
        raise ValueError("Caixa ou dimensões inválidas no COCO.")
    x, y, largura, altura = map(float, bbox)
    if not all(math.isfinite(v) for v in (x, y, largura, altura)) or largura <= 0 or altura <= 0:
        raise ValueError("Caixa COCO não finita ou sem área positiva.")
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(largura_imagem), x + largura), min(float(altura_imagem), y + altura)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Caixa COCO completamente fora da imagem. Revise a anotação original.")
    alterada = (x1, y1, x2, y2) != (x, y, x + largura, y + altura)
    return [(x1 + x2) / (2 * largura_imagem), (y1 + y2) / (2 * altura_imagem),
            (x2 - x1) / largura_imagem, (y2 - y1) / altura_imagem], alterada


def reconstruir_labels(pasta_dataset: str | Path, anotacoes: str | Path) -> dict:
    """Refaz labels da cópia usando caixas reais, sem limiar de frequência por categoria.

    Reutiliza a divisão recebida; não redistribui fotos entre partições.
    Labels existentes nessa cópia não são sobrescritos, para evitar mistura de políticas.
    """
    pasta = Path(pasta_dataset).resolve()
    caminho_json = Path(anotacoes)
    coco = ler_anotacoes(caminho_json)
    classes = categorias_selecionadas(coco)
    if not classes:
        raise ValueError("Nenhuma categoria da política foi encontrada. Confira o JSON.")
    imagens = {i["id"]: i for i in coco["images"]}
    por_imagem = defaultdict(list)
    contagens_categoria = Counter()
    for anotacao in coco["annotations"]:
        if anotacao["category_id"] in classes:
            por_imagem[anotacao["image_id"]].append(anotacao)
    registros = []
    ids_usados = set()
    for split in SPLITS:
        arquivos = sorted(p for p in (pasta / "images" / split).rglob("*")
                           if p.is_file() and p.suffix.lower() in EXTENSOES)
        if not arquivos:
            raise ValueError(f"Partição vazia: {split}.")
        if (pasta / "labels" / split).exists():
            raise FileExistsError("Já existem labels nesta cópia. Use uma nova pasta para converter novamente.")
        for arquivo in arquivos:
            encontrado = re.fullmatch(r"taco_(\d+)", arquivo.stem)
            if not encontrado or int(encontrado[1]) not in imagens:
                raise ValueError(f"Não consegui relacionar {arquivo.name} a uma imagem do COCO original.")
            imagem_id = int(encontrado[1])
            if imagem_id in ids_usados:
                raise ValueError(f"Imagem TACO repetida entre entradas/partições: {imagem_id}.")
            ids_usados.add(imagem_id)
            imagem = imagens[imagem_id]
            linhas = []
            for anotacao in por_imagem.get(imagem_id, []):
                caixa, ajustada = converter_caixa(anotacao["bbox"], imagem["width"], imagem["height"])
                linhas.append("0 " + " ".join(f"{v:.9f}" for v in caixa))
                contagens_categoria[anotacao["category_id"]] += 1
                registros.append({"split": split, "imagem_id_taco": imagem_id,
                                  "anotacao_id_coco": anotacao.get("id"),
                                  "categoria_original": classes[anotacao["category_id"]],
                                  "bbox_coco": anotacao["bbox"], "bbox_yolo": caixa,
                                  "interseccao_com_limites_da_imagem": ajustada})
            relativo = arquivo.relative_to(pasta / "images" / split).with_suffix(".txt")
            label = pasta / "labels" / split / relativo
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")
    yaml = pasta / "plastico.yaml"
    yaml.write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: plastico\n", encoding="utf-8")
    relatorio = {"origem": "Caixas reais de TACO/data/annotations.json fornecido pelo usuário",
                 "sha256_annotations_json": hashlib.sha256(caminho_json.read_bytes()).hexdigest(),
                 "classes_yolo": {0: "plastico"},
                 "categorias_originais_incluidas": [{"id": i, "nome": n, "caixas": contagens_categoria[i]}
                                                     for i, n in sorted(classes.items())],
                 "caixas": len(registros),
                 "caixas_interseccionadas_com_a_imagem": sum(r["interseccao_com_limites_da_imagem"] for r in registros),
                 "conversoes": registros,
                 "limitacoes": "A seleção de material e as anotações precisam de revisão visual humana. "
                               "Categorias ambíguas como Unlabeled litter e Rope & strings não são resolvidas automaticamente. "
                               "Esta conversão não comprova generalização para praias e rios."}
    (pasta / "conversao_taco.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    return relatorio
