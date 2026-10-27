"""Entrada e preparação de imagens. Contrato interno: uint8 em BGR."""

from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from numpy.typing import NDArray

Imagem = NDArray[np.uint8]
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 24_000_000


def validar_imagem(imagem: Imagem, *, colorida: bool = False) -> None:
    """Rejeita arrays vazios, tipos incompatíveis e dimensões inesperadas."""
    if not isinstance(imagem, np.ndarray) or imagem.size == 0:
        raise ValueError("A imagem está vazia ou é inválida.")
    if imagem.dtype != np.uint8:
        raise ValueError("A imagem deve usar pixels uint8 (0 a 255).")
    if imagem.ndim != 2 and not (imagem.ndim == 3 and imagem.shape[2] == 3):
        raise ValueError("Use uma imagem em cinza ou com três canais BGR.")
    if colorida and imagem.ndim != 3:
        raise ValueError("Esta operação requer uma imagem colorida BGR.")
    if imagem.shape[0] * imagem.shape[1] > MAX_PIXELS:
        raise ValueError("Imagem muito grande. Use no máximo 24 megapixels.")


def carregar_imagem(fonte: str | Path | bytes | BinaryIO) -> Imagem:
    """Decodifica JPG/PNG reais, incluindo caminhos com acentos no Windows.

    Retorna três canais BGR. O OpenCV aplica a orientação EXIF ao decodificar
    JPEG. PNG com transparência é composto sobre fundo branco explicitamente.
    """
    if isinstance(fonte, (str, Path)):
        caminho = Path(fonte)
        if not caminho.is_file():
            raise ValueError("Arquivo de imagem não encontrado.")
        if caminho.stat().st_size > MAX_BYTES:
            raise ValueError("Use uma imagem com até 20 MB.")
        dados = caminho.read_bytes()
    elif isinstance(fonte, bytes):
        dados = fonte
    else:
        dados = fonte.read(MAX_BYTES + 1)
    if not isinstance(dados, bytes) or not dados or len(dados) > MAX_BYTES:
        raise ValueError("Arquivo vazio, inválido ou maior que 20 MB.")
    png = dados.startswith(b"\x89PNG\r\n\x1a\n")
    jpeg = dados.startswith(b"\xff\xd8\xff")
    if not (png or jpeg):
        raise ValueError("O conteúdo do arquivo não é uma imagem JPG ou PNG.")
    # Checar dimensões do cabeçalho PNG antes de alocar o bitmap.
    if png and len(dados) >= 24:
        largura = int.from_bytes(dados[16:20], "big")
        altura = int.from_bytes(dados[20:24], "big")
        if largura * altura > MAX_PIXELS:
            raise ValueError("Imagem muito grande. Use no máximo 24 megapixels.")
    try:
        buffer = np.frombuffer(dados, dtype=np.uint8)
        imagem = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if imagem is None:
            raise ValueError("Não foi possível decodificar a imagem. Ela pode estar corrompida.")
        validar_imagem(imagem, colorida=True)
        if png:
            original = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
            if original is not None and original.ndim == 3 and original.shape[2] == 4:
                escala = 65535.0 if original.dtype == np.uint16 else 255.0
                alfa = original[:, :, 3:4].astype(np.float32) / escala
                cor = original[:, :, :3].astype(np.float32) * (255.0 / escala)
                imagem = np.rint(cor * alfa + 255 * (1 - alfa)).astype(np.uint8)
        return np.ascontiguousarray(imagem)
    except cv2.error as exc:
        raise ValueError("Não foi possível ler a imagem. Verifique o arquivo.") from exc


def redimensionar(imagem: Imagem, largura: int | None = None,
                  altura: int | None = None, *, ampliar: bool = False) -> Imagem:
    """Ajusta dentro dos limites dados, mantendo proporção e sem ampliar por padrão."""
    validar_imagem(imagem)
    limites = [valor for valor in (largura, altura) if valor is not None]
    if not limites or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in limites):
        raise ValueError("Informe largura e/ou altura inteiras e positivas.")
    h, w = imagem.shape[:2]
    fatores = []
    if largura is not None:
        fatores.append(largura / w)
    if altura is not None:
        fatores.append(altura / h)
    fator = min(fatores)
    if not ampliar:
        fator = min(fator, 1.0)
    tamanho = (max(1, round(w * fator)), max(1, round(h * fator)))
    metodo = cv2.INTER_AREA if fator < 1 else cv2.INTER_LINEAR
    return cv2.resize(imagem, tamanho, interpolation=metodo)


def padronizar_resolucao(imagem: Imagem, tamanho: tuple[int, int] = (640, 640)) -> Imagem:
    """Produz resolução exata (largura, altura) com bordas, sem deformar.

    Esta função é didática. O YOLO faz seu próprio letterbox e a inferência
    recebe a imagem original, preservando as coordenadas das caixas.
    """
    largura, altura = tamanho
    ajustada = redimensionar(imagem, largura, altura)
    h, w = ajustada.shape[:2]
    esquerda, topo = (largura - w) // 2, (altura - h) // 2
    cor = (114, 114, 114) if imagem.ndim == 3 else 114
    return cv2.copyMakeBorder(ajustada, topo, altura - h - topo,
                              esquerda, largura - w - esquerda,
                              cv2.BORDER_CONSTANT, value=cor)


def converter_cores(imagem: Imagem, destino: str = "RGB") -> Imagem:
    """Converte uma entrada BGR para RGB, GRAY ou HSV."""
    validar_imagem(imagem, colorida=True)
    codigos = {"RGB": cv2.COLOR_BGR2RGB, "GRAY": cv2.COLOR_BGR2GRAY, "HSV": cv2.COLOR_BGR2HSV}
    if destino.upper() not in codigos:
        raise ValueError("Espaço de cor inválido. Use RGB, GRAY ou HSV.")
    return cv2.cvtColor(imagem, codigos[destino.upper()])


def preparar_para_modelo(imagem: Imagem) -> Imagem:
    """Entrega uma cópia BGR contígua. YOLO normaliza e redimensiona internamente."""
    validar_imagem(imagem, colorida=True)
    return np.array(imagem, dtype=np.uint8, order="C", copy=True)
