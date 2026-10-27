"""Validação local do dataset YOLO e proveniência das execuções.

Nenhuma imagem ou anotação é criada por este módulo. As importações pesadas
só acontecem depois da verificação de configuração e de diretórios vazios.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
from uuid import uuid4

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
YAML_PADRAO = RAIZ_PROJETO / "dataset" / "plastico.yaml"
SPLITS = ("train", "val", "test")
EXTENSOES = {".jpg", ".jpeg", ".png"}


class ErroDataset(ValueError):
    """Configuração, imagens ou anotações do dataset inconsistentes."""


@dataclass(frozen=True)
class ConfiguracaoDataset:
    caminho_yaml: Path
    raiz: Path
    classes: dict[int, str]
    imagens: dict[str, Path]
    labels: dict[str, Path]


@dataclass(frozen=True)
class DatasetValidado:
    configuracao: ConfiguracaoDataset
    contagens: dict[str, dict[str, int]]
    manifesto: list[dict]
    sha256_configuracao: str
    sha256_conjunto: str

    def proveniencia(self) -> dict:
        """Identifica configuração, conteúdo e quantidade efetivamente validados."""
        return {
            "yaml_origem": str(self.configuracao.caminho_yaml),
            "sha256_yaml_origem": self.sha256_configuracao,
            "sha256_conjunto": self.sha256_conjunto,
            "classes": self.configuracao.classes,
            "contagens": self.contagens,
            "manifesto": self.manifesto,
        }


def _caminho_local(valor: object, base: Path, campo: str) -> Path:
    if not isinstance(valor, str) or not valor.strip() or "://" in valor:
        raise ErroDataset(f"O campo '{campo}' deve indicar um diretório local.")
    caminho = Path(valor).expanduser()
    return (base / caminho).resolve()


def carregar_configuracao(caminho_yaml: str | Path) -> ConfiguracaoDataset:
    """Lê YAML seguro; caminhos são relativos ao YAML, nunca à pasta corrente.

    Suporta a estrutura images/<partição> e labels/<partição>, incluindo
    subpastas correspondentes. Novas classes podem ser adicionadas no YAML.
    """
    caminho = Path(caminho_yaml).expanduser().resolve()
    if not caminho.is_file():
        raise ErroDataset(f"YAML do dataset não encontrado: {caminho}.")
    if caminho.suffix.lower() not in {".yaml", ".yml"}:
        raise ErroDataset("A configuração do dataset deve ser um arquivo .yaml ou .yml.")
    import yaml

    try:
        configuracao = yaml.safe_load(caminho.read_text(encoding="utf-8-sig"))
    except (yaml.YAMLError, UnicodeError) as exc:
        raise ErroDataset("YAML inválido. Verifique indentação, campos e codificação UTF-8.") from exc
    if not isinstance(configuracao, dict):
        raise ErroDataset("O YAML deve conter um mapeamento de campos.")
    permitidos = {"path", "train", "val", "test", "nc", "names"}
    extras = set(configuracao) - permitidos
    if extras:
        raise ErroDataset("Campos não permitidos no YAML: " + ", ".join(map(str, extras))
                          + ". Downloads e scripts automáticos não são aceitos.")
    faltantes = {"train", "val", "test", "nc", "names"} - set(configuracao)
    if faltantes:
        raise ErroDataset("Campos obrigatórios ausentes no YAML: " + ", ".join(sorted(faltantes)))
    nc = configuracao["nc"]
    nomes = configuracao["names"]
    if isinstance(nomes, list):
        nomes = dict(enumerate(nomes))
    if (type(nc) is not int or nc < 1 or not isinstance(nomes, dict)
            or any(type(chave) is not int for chave in nomes)
            or set(nomes) != set(range(nc))):
        raise ErroDataset("nc e names devem definir classes consecutivas a partir de 0.")
    if any(not isinstance(nome, str) or not nome.strip() for nome in nomes.values()):
        raise ErroDataset("Cada classe deve ter um nome de texto não vazio.")
    classes = {i: nomes[i].strip().lower() for i in range(nc)}
    if len(set(classes.values())) != nc:
        raise ErroDataset("Os nomes das classes devem ser distintos.")
    raiz = _caminho_local(configuracao.get("path", "."), caminho.parent, "path")
    imagens = {split: _caminho_local(configuracao[split], raiz, split) for split in SPLITS}
    labels = {}
    for split, pasta in imagens.items():
        if pasta.parent.name != "images":
            raise ErroDataset(f"{split}: use a estrutura images/<partição> e labels/<partição>.")
        labels[split] = pasta.parent.parent / "labels" / pasta.name
    return ConfiguracaoDataset(caminho, raiz, classes, imagens, labels)


def validar_label(caminho: Path, classes: dict[int, str]) -> int:
    """Valida cada linha classe xc yc largura altura, normalizada entre 0 e 1.

    Arquivo vazio representa uma imagem negativa revisada pelo anotador;
    arquivo ausente não é automaticamente interpretado como negativo.
    """
    try:
        linhas = caminho.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeError as exc:
        raise ErroDataset(f"Label {caminho.name}: use codificação UTF-8.") from exc
    quantidade = 0
    for numero, linha in enumerate(linhas, start=1):
        if not linha.strip():
            continue
        campos = linha.split()
        erro = f"Label {caminho.name}, linha {numero}"
        if len(campos) != 5:
            raise ErroDataset(f"{erro}: esperados 5 campos (classe xc yc largura altura).")
        try:
            classe = int(campos[0])
            xc, yc, largura, altura = (float(campo) for campo in campos[1:])
        except ValueError as exc:
            raise ErroDataset(f"{erro}: classe deve ser inteira e coordenadas numéricas.") from exc
        if classe not in classes:
            raise ErroDataset(f"{erro}: classe {classe} não existe no YAML.")
        valores = (xc, yc, largura, altura)
        if not all(math.isfinite(valor) for valor in valores):
            raise ErroDataset(f"{erro}: coordenadas devem ser finitas.")
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < largura <= 1 and 0 < altura <= 1):
            raise ErroDataset(f"{erro}: use coordenadas normalizadas e caixas com área positiva.")
        tolerancia = 1e-6  # Arredondamento decimal de exportadores de anotação.
        if (xc - largura / 2 < -tolerancia or yc - altura / 2 < -tolerancia
                or xc + largura / 2 > 1 + tolerancia or yc + altura / 2 > 1 + tolerancia):
            raise ErroDataset(f"{erro}: a caixa ultrapassa os limites da imagem.")
        quantidade += 1
    return quantidade


def _listar_imagens(pasta: Path) -> list[Path]:
    if not pasta.is_dir():
        return []
    return sorted(arquivo for arquivo in pasta.rglob("*")
                  if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES)


def validar_dataset(caminho_yaml: str | Path = YAML_PADRAO,
                    splits_obrigatorios: tuple[str, ...] = ("train", "val")) -> DatasetValidado:
    """Exige imagens/labels válidos e impede imagens idênticas entre partições.

    Todos os splits preenchidos são examinados; apenas os obrigatórios devem
    estar preenchidos e conter pelo menos um objeto anotado. SHA-256 identifica
    duplicatas byte a byte, mas não identifica recortes ou fotos quase iguais.
    """
    if not splits_obrigatorios or any(split not in SPLITS for split in splits_obrigatorios):
        raise ErroDataset("Escolha partições entre train, val e test.")
    config = carregar_configuracao(caminho_yaml)
    arquivos = {split: _listar_imagens(config.imagens[split]) for split in SPLITS}
    for split in splits_obrigatorios:
        if not arquivos[split]:
            raise ErroDataset(f"Dataset {split} inexistente ou vazio. Adicione fotografias JPG/PNG "
                              f"em {config.imagens[split]} e anote-as em {config.labels[split]}.")
    # Importação após detectar dataset vazio: nenhum modelo é carregado aqui.
    from processamento.preprocessamento import carregar_imagem
    from modelo.detector import sha256_arquivo

    manifesto: list[dict] = []
    contagens = {}
    hashes: dict[str, tuple[str, Path]] = {}
    for split in SPLITS:
        pasta, pasta_labels = config.imagens[split], config.labels[split]
        esperados: set[str] = set()
        stems: set[str] = set()
        objetos = negativos = 0
        for imagem in arquivos[split]:
            relativo = imagem.relative_to(pasta)
            stem = imagem.stem.casefold()
            if stem in stems:
                raise ErroDataset(f"{split}: nomes-base colidem ({imagem.name}). Renomeie imagem e label.")
            stems.add(stem)
            label_relativo = relativo.with_suffix(".txt")
            esperados.add(label_relativo.as_posix().casefold())
            label = pasta_labels / label_relativo
            if not label.is_file():
                raise ErroDataset(f"Label ausente para {split}/{relativo.as_posix()}. "
                                  "Anote a imagem; use .txt vazio somente para negativos revisados.")
            try:
                carregar_imagem(imagem)
            except (ValueError, OSError) as exc:
                raise ErroDataset(f"Imagem inválida em {split}/{relativo.as_posix()}: {exc}") from exc
            quantidade = validar_label(label, config.classes)
            objetos += quantidade
            negativos += quantidade == 0
            resumo = sha256_arquivo(imagem)
            if resumo in hashes and hashes[resumo][0] != split:
                anterior_split, anterior = hashes[resumo]
                raise ErroDataset(f"Vazamento de dados: {split}/{relativo.as_posix()} é idêntica "
                                  f"a {anterior_split}/{anterior.name}. Separe as partições.")
            hashes[resumo] = (split, imagem)
            manifesto.append({"split": split, "imagem": relativo.as_posix(),
                              "sha256_imagem": resumo, "sha256_label": sha256_arquivo(label),
                              "objetos": quantidade})
        existentes = {arquivo.relative_to(pasta_labels).as_posix().casefold()
                      for arquivo in pasta_labels.rglob("*")
                      if arquivo.is_file() and arquivo.suffix.lower() == ".txt"}
        orfaos = existentes - esperados
        if orfaos:
            raise ErroDataset(f"{split}: labels sem imagem correspondente: " + ", ".join(sorted(orfaos)[:5]))
        if split in splits_obrigatorios and objetos == 0:
            raise ErroDataset(f"{split}: nenhuma anotação positiva. Inclua imagens com plástico "
                              "visível e caixas revisadas; negativos sozinhos não permitem esta avaliação.")
        contagens[split] = {"imagens": len(arquivos[split]), "objetos": objetos,
                            "negativos": int(negativos)}
    serializado = json.dumps(manifesto, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return DatasetValidado(config, contagens, manifesto, sha256_arquivo(config.caminho_yaml),
                           hashlib.sha256(serializado).hexdigest())


def salvar_yaml_normalizado(dataset: DatasetValidado, destino: Path) -> Path:
    """Escreve config de execução separada, com caminhos locais absolutos."""
    import yaml

    config = dataset.configuracao
    conteudo = {"path": str(config.raiz),
                **{split: str(config.imagens[split]) for split in SPLITS},
                "nc": len(config.classes), "names": config.classes}
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("x", encoding="utf-8") as arquivo:
        yaml.safe_dump(conteudo, arquivo, allow_unicode=True, sort_keys=False)
    return destino.resolve()


def novo_diretorio(base: Path, prefixo: str) -> Path:
    """Cria pasta exclusiva para não substituir resultados anteriores."""
    identificador = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destino = base / f"{prefixo}_{identificador}_{uuid4().hex[:8]}"
    destino.mkdir(parents=True, exist_ok=False)
    return destino


def versoes_ambiente() -> dict[str, str]:
    versoes = {"python": platform.python_version()}
    for pacote in ("ultralytics", "torch", "numpy", "opencv-python", "PyYAML"):
        try:
            versoes[pacote] = metadata.version(pacote)
        except metadata.PackageNotFoundError:
            versoes[pacote] = "não instalado"
    return versoes


def salvar_json_exclusivo(destino: Path, conteudo: dict) -> None:
    """Valida JSON antes de abrir destino, sem aceitar NaN ou sobrescrita."""
    texto = json.dumps(conteudo, ensure_ascii=False, indent=2, allow_nan=False)
    with destino.open("x", encoding="utf-8") as arquivo:
        arquivo.write(texto + "\n")
