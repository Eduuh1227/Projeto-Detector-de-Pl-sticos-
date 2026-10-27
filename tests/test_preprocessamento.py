"""Testes com matrizes/arquivos temporários, nunca com dados de treinamento inventados."""

from io import BytesIO

import cv2
import numpy as np
import pytest

from processamento.filtros import filtro_gaussiano, filtro_mediana
from processamento.preprocessamento import (
    carregar_imagem, converter_cores, padronizar_resolucao,
    preparar_para_modelo, redimensionar, validar_imagem,
)
from processamento.segmentacao import segmentar


def test_leitura_png_valida_caminho_com_acentos(tmp_path, imagem_bgr, png_bytes):
    caminho = tmp_path / "imagem técnica.png"
    caminho.write_bytes(png_bytes)
    assert np.array_equal(carregar_imagem(caminho), imagem_bgr)


def test_leitura_bytes_e_stream(imagem_bgr, png_bytes):
    assert np.array_equal(carregar_imagem(png_bytes), imagem_bgr)
    assert np.array_equal(carregar_imagem(BytesIO(png_bytes)), imagem_bgr)


def test_leitura_jpeg(imagem_bgr):
    sucesso, buffer = cv2.imencode(".jpg", imagem_bgr)
    assert sucesso
    imagem = carregar_imagem(buffer.tobytes())
    assert imagem.shape == imagem_bgr.shape
    assert imagem.dtype == np.uint8


@pytest.mark.parametrize("dados", [b"", b"texto renomeado.jpg", b"\xff\xd8\xff\x00", b"\x89PNG\r\n\x1a\n"])
def test_arquivo_invalido_ou_corrompido(dados):
    with pytest.raises(ValueError):
        carregar_imagem(dados)


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(ValueError, match="não encontrado"):
        carregar_imagem(tmp_path / "ausente.jpg")


def test_transparencia_png_com_fundo_branco():
    pixels = np.array([[[20, 40, 80, 0], [20, 40, 80, 255]]], dtype=np.uint8)
    sucesso, buffer = cv2.imencode(".png", pixels)
    assert sucesso
    lida = carregar_imagem(buffer.tobytes())
    assert lida[0, 0].tolist() == [255, 255, 255]
    assert lida[0, 1].tolist() == [20, 40, 80]


def test_redimensionamento_preserva_proporcao_e_original(imagem_bgr):
    antes = imagem_bgr.copy()
    assert redimensionar(imagem_bgr, largura=20, altura=20).shape == (10, 20, 3)
    assert redimensionar(imagem_bgr, altura=10).shape == (10, 20, 3)
    assert redimensionar(imagem_bgr, largura=160).shape == imagem_bgr.shape
    assert redimensionar(imagem_bgr, largura=160, ampliar=True).shape == (80, 160, 3)
    assert np.array_equal(imagem_bgr, antes)


@pytest.mark.parametrize("largura", [0, -1, 1.5, True])
def test_redimensionamento_rejeita_limites_invalidos(imagem_bgr, largura):
    with pytest.raises(ValueError):
        redimensionar(imagem_bgr, largura=largura)


def test_padronizacao_adiciona_bordas_sem_deformar(imagem_bgr):
    ajustada = padronizar_resolucao(imagem_bgr, (80, 80))
    assert ajustada.shape == (80, 80, 3)
    assert np.array_equal(ajustada[20:60], imagem_bgr)
    assert np.all(ajustada[:20] == 114)


def test_bgr_para_rgb_e_cinza(imagem_bgr):
    assert converter_cores(imagem_bgr)[0, 0].tolist() == [140, 70, 20]
    assert converter_cores(imagem_bgr, "GRAY").shape == imagem_bgr.shape[:2]
    assert converter_cores(imagem_bgr, "HSV").shape == imagem_bgr.shape
    with pytest.raises(ValueError):
        converter_cores(imagem_bgr, "INVALIDO")


@pytest.mark.parametrize("invalida", [None, np.array([], dtype=np.uint8),
                                       np.zeros((3, 3), dtype=float), np.zeros((3, 3, 4), dtype=np.uint8)])
def test_validacao_de_array(invalida):
    with pytest.raises(ValueError):
        validar_imagem(invalida)


def test_preparacao_do_modelo_nao_altera_resolucao_cores_ou_original(imagem_bgr):
    preparada = preparar_para_modelo(imagem_bgr[:, ::-1])
    assert preparada.flags.c_contiguous
    assert np.array_equal(preparada, imagem_bgr[:, ::-1])
    assert not np.shares_memory(preparada, imagem_bgr)


@pytest.mark.parametrize("filtro", [filtro_gaussiano, filtro_mediana])
def test_filtros_suavizam_impulso_sem_alterar_entrada(filtro):
    matriz = np.zeros((15, 15, 3), dtype=np.uint8)
    matriz[7, 7] = 255
    antes = matriz.copy()
    filtrada = filtro(matriz, kernel=3)
    assert filtrada.shape == matriz.shape
    assert filtrada.dtype == np.uint8
    assert filtrada[7, 7, 0] < matriz[7, 7, 0]
    assert np.array_equal(matriz, antes)


@pytest.mark.parametrize("filtro", [filtro_gaussiano, filtro_mediana])
@pytest.mark.parametrize("kernel", [0, 1, 2, 4, -3, 3.5, True])
def test_kernels_invalidos(imagem_bgr, filtro, kernel):
    with pytest.raises(ValueError):
        filtro(imagem_bgr, kernel=kernel)


@pytest.mark.parametrize("sigma", [-1, float("nan"), float("inf")])
def test_sigma_invalido(imagem_bgr, sigma):
    with pytest.raises(ValueError):
        filtro_gaussiano(imagem_bgr, sigma=sigma)


def test_limiarizacao_manual_e_otsu():
    matriz = np.zeros((20, 20), dtype=np.uint8)
    matriz[:, 10:] = 200
    manual, limiar = segmentar(matriz, limiar=100, otsu=False)
    otsu, automatico = segmentar(matriz, otsu=True)
    assert limiar == 100
    assert 0 <= automatico < 200
    assert np.array_equal(manual, otsu)
    assert np.all(manual[:, :10] == 0)
    assert np.all(manual[:, 10:] == 255)


def test_abertura_remove_pixel_isolado():
    matriz = np.zeros((15, 15), dtype=np.uint8)
    matriz[7, 7] = 255
    sem, _ = segmentar(matriz, otsu=False)
    abertura, _ = segmentar(matriz, otsu=False, morfologia="abertura")
    assert sem.sum() == 255
    assert abertura.sum() == 0


def test_segmentacao_nao_muta_entrada(imagem_bgr):
    antes = imagem_bgr.copy()
    mascara, _ = segmentar(imagem_bgr, morfologia="fechamento")
    assert set(np.unique(mascara)).issubset({0, 255})
    assert np.array_equal(imagem_bgr, antes)
    with pytest.raises(ValueError):
        segmentar(imagem_bgr, limiar=256)
    with pytest.raises(ValueError):
        segmentar(imagem_bgr, morfologia="invalida")
