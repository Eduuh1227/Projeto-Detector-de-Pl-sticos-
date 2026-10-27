"""Filtros para comparação acadêmica; não classificam materiais."""

import cv2

from .preprocessamento import Imagem, validar_imagem


def validar_kernel(kernel: int) -> None:
    if not isinstance(kernel, int) or isinstance(kernel, bool) or kernel < 3 or kernel % 2 == 0:
        raise ValueError("O tamanho do kernel deve ser um inteiro ímpar maior ou igual a 3.")


def filtro_gaussiano(imagem: Imagem, kernel: int = 5, sigma: float = 0) -> Imagem:
    """Suaviza a imagem com kernel Gaussiano; sigma=0 usa a estimativa do OpenCV."""
    validar_imagem(imagem)
    validar_kernel(kernel)
    if not 0 <= sigma < float("inf"):
        raise ValueError("Sigma deve ser finito e não negativo.")
    return cv2.GaussianBlur(imagem, (kernel, kernel), sigmaX=sigma)


def filtro_mediana(imagem: Imagem, kernel: int = 5) -> Imagem:
    """Substitui cada pixel pela mediana da vizinhança configurada."""
    validar_imagem(imagem)
    validar_kernel(kernel)
    return cv2.medianBlur(imagem, kernel)
