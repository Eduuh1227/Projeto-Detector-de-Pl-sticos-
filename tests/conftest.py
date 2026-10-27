"""Fixtures técnicas temporárias; não são fotografias, dataset ou evidência científica."""

import cv2
import numpy as np
import pytest


@pytest.fixture
def imagem_bgr():
    """Matriz de pixels apenas para testar contratos de processamento."""
    imagem = np.zeros((40, 80, 3), dtype=np.uint8)
    imagem[:, :40] = (20, 70, 140)
    imagem[:, 40:] = (210, 160, 90)
    imagem[20, 20] = (255, 255, 255)
    return imagem


@pytest.fixture
def png_bytes(imagem_bgr):
    sucesso, buffer = cv2.imencode(".png", imagem_bgr)
    assert sucesso
    return buffer.tobytes()
