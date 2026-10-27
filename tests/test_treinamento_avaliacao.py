"""Contratos de treino/avaliação sem pesos, treino ou resultados científicos.

Doubles são usados exclusivamente para verificar rejeição de falhas e extração
de campos. Nenhum resultado desses testes representa desempenho do projeto.
"""

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from modelo import avaliacao, detector, treinamento
from modelo.dados import ErroDataset


@pytest.fixture
def yaml_vazio(tmp_path):
    raiz = tmp_path / "vazio"
    for split in ("train", "val", "test"):
        (raiz / "images" / split).mkdir(parents=True)
        (raiz / "labels" / split).mkdir(parents=True)
    caminho = raiz / "plastico.yaml"
    caminho.write_text(yaml.safe_dump({"path": ".", "train": "images/train",
                                      "val": "images/val", "test": "images/test",
                                      "nc": 1, "names": {0: "plastico"}}), encoding="utf-8")
    return caminho


@pytest.fixture
def proibir_yolo(monkeypatch):
    """Qualquer tentativa de importar o modelo falha imediatamente no teste."""
    importar = builtins.__import__

    def sem_yolo(nome, *args, **kwargs):
        if nome == "ultralytics" or nome.startswith("ultralytics."):
            raise AssertionError("Dataset inválido não deve iniciar YOLO.")
        return importar(nome, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", sem_yolo)


def test_treino_vazio_nao_carrega_modelo_nem_cria_saidas(yaml_vazio, tmp_path, proibir_yolo):
    execucoes, modelos = tmp_path / "execucoes", tmp_path / "pesos"
    with pytest.raises(ErroDataset, match="vazio"):
        treinamento.treinar(tmp_path / "ausente.pt", dataset=yaml_vazio,
                            pasta_execucoes=execucoes, pasta_modelos=modelos)
    assert not execucoes.exists()
    assert not modelos.exists()


@pytest.mark.parametrize("split", ["val", "test"])
def test_avaliacao_vazia_nao_carrega_modelo_nem_inventa_metricas(yaml_vazio, tmp_path, proibir_yolo, split):
    destino = tmp_path / "metricas"
    with pytest.raises(ErroDataset, match="vazio"):
        avaliacao.avaliar(tmp_path / "ausente.pt", dataset=yaml_vazio,
                          split=split, pasta_metricas=destino)
    assert not destino.exists()


def test_cli_treino_vazio_exibe_mensagem_amigavel(yaml_vazio, tmp_path, capsys, proibir_yolo):
    codigo = treinamento.main(["--dataset", str(yaml_vazio), "--pesos-base", str(tmp_path / "ausente.pt")])
    capturado = capsys.readouterr()
    assert codigo == 1
    assert "Adicione fotografias" in capturado.err
    assert "Melhor modelo salvo" not in capturado.out
    assert "Traceback" not in capturado.err


def test_cli_avaliacao_vazia_nao_afirma_ter_metricas(yaml_vazio, tmp_path, capsys, proibir_yolo):
    codigo = avaliacao.main(["--dataset", str(yaml_vazio), "--pesos", str(tmp_path / "ausente.pt")])
    capturado = capsys.readouterr()
    assert codigo == 1
    assert "Adicione fotografias" in capturado.err
    assert "Métricas reais e proveniência salvas" not in capturado.out
    assert "Traceback" not in capturado.err


@pytest.mark.parametrize("resultado", [
    None,
    SimpleNamespace(),
    SimpleNamespace(box=SimpleNamespace(mp=0.2, mr=0.3, map50=0.4)),
    SimpleNamespace(box=SimpleNamespace(mp=None, mr=0.3, map50=0.4, map=0.1)),
])
def test_metricas_ausentes_nao_sao_substituidas_por_zero(resultado):
    with pytest.raises(RuntimeError, match="Nenhum arquivo de métricas"):
        avaliacao.extrair_metricas(resultado)


@pytest.mark.parametrize("valor", [float("nan"), float("inf"), -0.1, 1.1])
def test_metricas_fora_do_dominio_sao_rejeitadas(valor):
    double = SimpleNamespace(box=SimpleNamespace(mp=valor, mr=0.3, map50=0.4, map=0.1))
    with pytest.raises(RuntimeError, match="métricas inválidas"):
        avaliacao.extrair_metricas(double)


def test_extracao_preserva_campos_do_double_sem_gerar_relatorio():
    """Números de contrato em memória, não medições nem resultados do projeto."""
    double = SimpleNamespace(box=SimpleNamespace(mp=0.11, mr=0.22, map50=0.33, map=0.44))
    assert avaliacao.extrair_metricas(double) == {
        "precision": 0.11, "recall": 0.22, "mAP50": 0.33, "mAP50-95": 0.44}


@pytest.mark.parametrize("falha_externa", [False, True])
def test_avaliacao_sem_retorno_valido_nao_grava_metricas(monkeypatch, tmp_path, falha_externa):
    """Isola a fronteira YOLO; só pode sobrar YAML técnico temporário, nunca métricas."""
    dados_double = SimpleNamespace(configuracao=SimpleNamespace(classes={0: "plastico"}))
    monkeypatch.setattr(avaliacao, "validar_dataset", lambda *args: dados_double)

    def yaml_tecnico(dados, caminho):
        caminho.write_text("# Configuração temporária de teste unitário\n", encoding="utf-8")
        return caminho

    monkeypatch.setattr(avaliacao, "salvar_yaml_normalizado", yaml_tecnico)

    def validar_double(**kwargs):
        assert kwargs["split"] == "test"
        assert Path(kwargs["data"]).is_file()
        if falha_externa:
            raise RuntimeError("Falha simulada da dependência, sem inferência.")
        return SimpleNamespace()  # Sem métricas, nunca deve virar zeros.

    objeto = SimpleNamespace(modelo=SimpleNamespace(val=validar_double))
    monkeypatch.setattr(detector, "DetectorPlastico", lambda *args, **kwargs: objeto)
    destino = tmp_path / "metricas"
    with pytest.raises(detector.ErroModelo, match="Nenhum JSON de métricas"):
        avaliacao.avaliar(tmp_path / "marcador.pt", pasta_metricas=destino)
    assert not list(destino.rglob("metricas.json"))


def test_treino_sem_pesos_nao_carrega_yolo_apos_validar_dados(monkeypatch, tmp_path, proibir_yolo):
    monkeypatch.setattr(treinamento, "validar_dataset", lambda *args: object())
    destino = tmp_path / "treinos"
    with pytest.raises(detector.ErroModelo, match="não encontrado"):
        treinamento.treinar(tmp_path / "ausente.pt", pasta_execucoes=destino,
                            pasta_modelos=tmp_path / "modelos")
    assert not destino.exists()
