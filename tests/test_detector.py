"""Doubles testam o contrato do adaptador, não detecção real nem desempenho científico."""

import builtins
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from modelo import detector as modulo
from modelo.detector import DetectorPlastico, ErroModelo, carregar_yolo_local, verificar_pesos


class CaixasDouble:
    def __init__(self, caixas=(), scores=(), classes=()):
        self.xyxy = np.array(caixas, dtype=float).reshape(-1, 4)
        self.conf = np.array(scores, dtype=float)
        self.cls = np.array(classes, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self


class ModeloDouble:
    """Não carrega PyTorch ou YOLO; só substitui a fronteira externa no teste."""
    names = {0: "plastico"}
    task = "detect"

    def __init__(self, caixas=None):
        self.caixas = CaixasDouble() if caixas is None else caixas
        self.argumentos = None

    def predict(self, **kwargs):
        self.argumentos = kwargs
        return [SimpleNamespace(boxes=self.caixas)]


@pytest.fixture
def pesos_dummy(tmp_path):
    """Arquivo marcador temporário, nunca tratado como checkpoint real."""
    caminho = tmp_path / "marcador_teste.pt"
    caminho.write_bytes(b"double-de-teste-sem-pesos-reais")
    return caminho


def criar_detector(monkeypatch, pesos, modelo):
    monkeypatch.setattr(modulo, "carregar_yolo_local", lambda _: modelo)
    return DetectorPlastico(pesos)


def test_modelo_ausente_nao_importa_ultralytics(monkeypatch, tmp_path):
    original = builtins.__import__

    def bloquear(nome, *args, **kwargs):
        if nome == "ultralytics" or nome.startswith("ultralytics."):
            raise AssertionError("YOLO não deve ser importado sem arquivo de pesos.")
        return original(nome, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", bloquear)
    with pytest.raises(ErroModelo, match="não encontrado"):
        DetectorPlastico(tmp_path / "ausente.pt")


def test_rejeita_arquivo_vazio_ou_extensao_invalida(tmp_path):
    vazio = tmp_path / "vazio.pt"
    vazio.touch()
    errado = tmp_path / "invalido.txt"
    errado.write_text("marcador", encoding="utf-8")
    for caminho in (vazio, errado):
        with pytest.raises(ErroModelo):
            verificar_pesos(caminho)


def test_modelo_generico_rejeitado(monkeypatch, pesos_dummy):
    modelo = ModeloDouble()
    modelo.names = {0: "person", 1: "bicycle"}
    with pytest.raises(ErroModelo, match="Classes incompatíveis"):
        criar_detector(monkeypatch, pesos_dummy, modelo)


def test_loader_rejeita_tarefa_incorreta(monkeypatch, pesos_dummy):
    fake = SimpleNamespace(YOLO=lambda *args, **kwargs: SimpleNamespace(task="classify"))
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.checks",
                        SimpleNamespace(check_yolov5u_filename=lambda nome, **kwargs: nome))
    with pytest.raises(ErroModelo, match="detecção de objetos"):
        carregar_yolo_local(pesos_dummy)


def test_loader_traduz_falha_de_pesos(monkeypatch, pesos_dummy):
    def falhar(*args, **kwargs):
        raise ValueError("checkpoint inválido")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=falhar))
    monkeypatch.setitem(sys.modules, "ultralytics.utils.checks",
                        SimpleNamespace(check_yolov5u_filename=lambda nome, **kwargs: nome))
    with pytest.raises(ErroModelo, match="Falha ao carregar"):
        carregar_yolo_local(pesos_dummy)


def test_loader_traduz_biblioteca_ausente(monkeypatch, pesos_dummy):
    original = builtins.__import__

    def ausente(nome, *args, **kwargs):
        if nome == "ultralytics":
            raise ModuleNotFoundError("Biblioteca indisponível no teste")
        return original(nome, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", ausente)
    with pytest.raises(ErroModelo, match="indisponível"):
        carregar_yolo_local(pesos_dummy)


def test_loader_nao_busca_pesos_com_nome_transformado(monkeypatch, pesos_dummy):
    def nao_deve_carregar(*args, **kwargs):
        raise AssertionError("Um caminho alterado não pode chegar ao carregador YOLO.")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=nao_deve_carregar))
    monkeypatch.setitem(sys.modules, "ultralytics.utils.checks",
                        SimpleNamespace(check_yolov5u_filename=lambda nome, **kwargs: nome + "outro"))
    with pytest.raises(ErroModelo, match="alteraria o nome"):
        carregar_yolo_local(pesos_dummy)


def test_metadados_de_classes_invalidos(monkeypatch, pesos_dummy):
    modelo = ModeloDouble()
    modelo.names = None
    with pytest.raises(ErroModelo, match="metadados de classes"):
        criar_detector(monkeypatch, pesos_dummy, modelo)


def test_sem_deteccoes_preserva_imagem(monkeypatch, pesos_dummy, imagem_bgr):
    detector = criar_detector(monkeypatch, pesos_dummy, ModeloDouble())
    resultado = detector.inferir(imagem_bgr)
    assert resultado.deteccoes == []
    assert np.array_equal(resultado.imagem_bgr, imagem_bgr)
    assert not np.shares_memory(resultado.imagem_bgr, imagem_bgr)


def test_caixa_na_borda_e_metadados_vem_do_double(monkeypatch, pesos_dummy, imagem_bgr):
    altura, largura = imagem_bgr.shape[:2]
    modelo = ModeloDouble(CaixasDouble([[largura - 1, altura - 1, largura, altura]], [0.71], [0]))
    detector = criar_detector(monkeypatch, pesos_dummy, modelo)
    antes = imagem_bgr.copy()
    resultado = detector.inferir(imagem_bgr, confianca=0.4, iou=0.5, imgsz=320)
    assert len(resultado.deteccoes) == 1
    item = resultado.deteccoes[0]
    assert item.caixa_xyxy == (largura - 1, altura - 1, largura, altura)
    assert item.confianca == 0.71
    assert item.classe == "plastico"
    assert np.array_equal(imagem_bgr, antes)
    assert modelo.argumentos["conf"] == 0.4
    assert modelo.argumentos["iou"] == 0.5
    assert modelo.argumentos["imgsz"] == 320
    assert modelo.argumentos["save"] is False
    assert np.array_equal(modelo.argumentos["source"], imagem_bgr)
    assert not np.shares_memory(modelo.argumentos["source"], imagem_bgr)
    assert resultado.metadados["resolucao_original"] == [largura, altura]
    assert len(resultado.metadados["sha256_pesos"]) == 64


@pytest.mark.parametrize("scores,classes", [([float("nan")], [0]), ([1.1], [0]), ([0.4], [1]), ([0.4], [0.5])])
def test_rejeita_saida_invalida(monkeypatch, pesos_dummy, imagem_bgr, scores, classes):
    modelo = ModeloDouble(CaixasDouble([[1, 1, 5, 5]], scores, classes))
    detector = criar_detector(monkeypatch, pesos_dummy, modelo)
    with pytest.raises(ErroModelo):
        detector.inferir(imagem_bgr)


@pytest.mark.parametrize("parametros", [{"confianca": 0}, {"confianca": float("nan")},
                                         {"iou": 1.1}, {"imgsz": 33}])
def test_rejeita_parametros_invalidos(monkeypatch, pesos_dummy, imagem_bgr, parametros):
    modelo = ModeloDouble()
    detector = criar_detector(monkeypatch, pesos_dummy, modelo)
    with pytest.raises(ValueError):
        detector.inferir(imagem_bgr, **parametros)
    assert modelo.argumentos is None


def test_falha_na_inferencia_e_traduzida(monkeypatch, pesos_dummy, imagem_bgr):
    modelo = ModeloDouble()

    def falhar(**kwargs):
        raise RuntimeError("falha externa simulada")

    modelo.predict = falhar
    detector = criar_detector(monkeypatch, pesos_dummy, modelo)
    with pytest.raises(ErroModelo, match="Falha na inferência"):
        detector.inferir(imagem_bgr)
