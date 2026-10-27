"""Demonstração de limiarização. A máscara não identifica plástico."""

import cv2

from .filtros import validar_kernel
from .preprocessamento import Imagem, converter_cores, validar_imagem


def segmentar(imagem: Imagem, limiar: int = 127, *, otsu: bool = True,
              morfologia: str = "nenhuma", kernel: int = 3) -> tuple[Imagem, float]:
    """Retorna máscara binária e limiar utilizado; aceita BGR ou escala de cinza."""
    validar_imagem(imagem)
    if not isinstance(limiar, int) or not 0 <= limiar <= 255:
        raise ValueError("O limiar deve ser um inteiro entre 0 e 255.")
    cinza = converter_cores(imagem, "GRAY") if imagem.ndim == 3 else imagem.copy()
    modo = cv2.THRESH_BINARY | (cv2.THRESH_OTSU if otsu else 0)
    usado, mascara = cv2.threshold(cinza, 0 if otsu else limiar, 255, modo)
    operacoes = {"abertura": cv2.MORPH_OPEN, "fechamento": cv2.MORPH_CLOSE}
    if morfologia != "nenhuma":
        if morfologia not in operacoes:
            raise ValueError("Morfologia deve ser nenhuma, abertura ou fechamento.")
        validar_kernel(kernel)
        elemento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        mascara = cv2.morphologyEx(mascara, operacoes[morfologia], elemento)
    return mascara, float(usado)
