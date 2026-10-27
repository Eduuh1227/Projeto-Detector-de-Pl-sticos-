"""Validação com fixtures técnicas em tmp_path, nunca com dados científicos.

As pequenas matrizes/linhas abaixo servem apenas para exercitar formatos de
arquivo. Não são fotografias, anotações de resíduos ou amostras de treinamento.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from modelo.dados import (ErroDataset, carregar_configuracao, salvar_yaml_normalizado,
                          validar_dataset, validar_label)


@pytest.fixture
def dataset_tecnico(tmp_path):
    """Cria apenas diretórios e configuração temporários; imagens vêm por chamada."""
    raiz = tmp_path / "dados_teste"
    for split in ("train", "val", "test"):
        (raiz / "images" / split).mkdir(parents=True)
        (raiz / "labels" / split).mkdir(parents=True)
    configuracao = {"path": ".", "train": "images/train", "val": "images/val",
                    "test": "images/test", "nc": 1, "names": {0: "plastico"}}
    arquivo = raiz / "plastico.yaml"
    arquivo.write_text(yaml.safe_dump(configuracao), encoding="utf-8")

    def adicionar(split: str, nome: str = "matriz", valor: int = 10,
                  label: str | None = "0 0.5 0.5 0.5 0.5\n") -> Path:
        imagem = raiz / "images" / split / f"{nome}.png"
        imagem.parent.mkdir(parents=True, exist_ok=True)
        ok, dados = cv2.imencode(".png", np.full((8, 12, 3), valor, dtype=np.uint8))
        assert ok
        imagem.write_bytes(dados.tobytes())
        if label is not None:
            anotacao = raiz / "labels" / split / f"{nome}.txt"
            anotacao.parent.mkdir(parents=True, exist_ok=True)
            anotacao.write_text(label, encoding="utf-8")
        return imagem

    return arquivo, adicionar


def test_caminhos_sao_relativos_ao_yaml(dataset_tecnico, monkeypatch, tmp_path):
    arquivo, _ = dataset_tecnico
    outra_pasta = tmp_path / "outra"
    outra_pasta.mkdir()
    monkeypatch.chdir(outra_pasta)
    config = carregar_configuracao(arquivo)
    assert config.raiz == arquivo.parent
    assert config.imagens["train"] == arquivo.parent / "images" / "train"
    assert config.labels["test"] == arquivo.parent / "labels" / "test"


@pytest.mark.parametrize("conteudo", ["names: [", "- um\n- dois", "", "!!python/object:builtins.object {}"])
def test_yaml_invalido_ou_nao_mapeamento_e_rejeitado(tmp_path, conteudo):
    arquivo = tmp_path / "invalido.yaml"
    arquivo.write_text(conteudo, encoding="utf-8")
    with pytest.raises(ErroDataset):
        carregar_configuracao(arquivo)


@pytest.mark.parametrize("mudanca", [
    {"download": "comando_nao_deve_executar"},
    {"train": "https://example.invalid/dados"},
    {"nc": 2},
    {"names": {1: "plastico"}},
    {"names": {0: ""}},
])
def test_yaml_inseguro_ou_inconsistente_e_rejeitado(dataset_tecnico, mudanca):
    arquivo, _ = dataset_tecnico
    conteudo = yaml.safe_load(arquivo.read_text(encoding="utf-8"))
    conteudo.update(mudanca)
    arquivo.write_text(yaml.safe_dump(conteudo), encoding="utf-8")
    with pytest.raises(ErroDataset):
        carregar_configuracao(arquivo)


def test_dataset_vazio_exige_fotos_e_anotacoes(dataset_tecnico):
    arquivo, _ = dataset_tecnico
    with pytest.raises(ErroDataset, match="inexistente ou vazio.*Adicione fotografias"):
        validar_dataset(arquivo)


def test_yaml_ausente_e_rejeitado(tmp_path):
    with pytest.raises(ErroDataset, match="não encontrado"):
        validar_dataset(tmp_path / "ausente.yaml")


def test_label_ausente_nao_e_tratado_como_negativo(dataset_tecnico):
    arquivo, adicionar = dataset_tecnico
    adicionar("train", label=None)
    with pytest.raises(ErroDataset, match="Label ausente"):
        validar_dataset(arquivo, ("train",))


def test_imagem_corrompida_e_rejeitada(dataset_tecnico):
    arquivo, adicionar = dataset_tecnico
    imagem = adicionar("train")
    imagem.write_bytes(b"arquivo corrompido, nao e PNG")
    with pytest.raises(ErroDataset, match="Imagem inválida"):
        validar_dataset(arquivo, ("train",))


def test_label_orfao_e_rejeitado(dataset_tecnico):
    arquivo, adicionar = dataset_tecnico
    adicionar("train")
    (arquivo.parent / "labels/train/orfao.txt").write_text("", encoding="utf-8")
    with pytest.raises(ErroDataset, match="sem imagem correspondente"):
        validar_dataset(arquivo, ("train",))


@pytest.mark.parametrize("linha", [
    "0 0.5 0.5 0.2",                 # Número de colunas incorreto.
    "1 0.5 0.5 0.2 0.2",             # Classe não definida.
    "0.5 0.5 0.5 0.2 0.2",           # Classe não inteira.
    "0 nan 0.5 0.2 0.2",             # Não finito.
    "0 0.5 0.5 inf 0.2",             # Não finito.
    "0 0.5 0.5 0 0.2",               # Sem área.
    "0 1.5 0.5 0.2 0.2",             # Centro fora da imagem.
    "0 0.1 0.5 0.5 0.2",             # Caixa ultrapassa borda.
])
def test_label_invalido_e_rejeitado(tmp_path, linha):
    arquivo = tmp_path / "label_formato.txt"
    arquivo.write_text(linha, encoding="utf-8")
    with pytest.raises(ErroDataset):
        validar_label(arquivo, {0: "plastico"})


def test_negativo_vazio_e_aceito_mas_nao_substitui_positivos(dataset_tecnico):
    arquivo, adicionar = dataset_tecnico
    adicionar("train", label="")
    with pytest.raises(ErroDataset, match="nenhuma anotação positiva"):
        validar_dataset(arquivo, ("train",))
    adicionar("train", nome="positivo_formato", valor=20)
    validado = validar_dataset(arquivo, ("train",))
    assert validado.contagens["train"] == {"imagens": 2, "objetos": 1, "negativos": 1}


def test_duplicata_entre_particoes_e_rejeitada(dataset_tecnico):
    arquivo, adicionar = dataset_tecnico
    adicionar("train", nome="treino", valor=12)
    adicionar("val", nome="validacao", valor=12)
    with pytest.raises(ErroDataset, match="Vazamento de dados"):
        validar_dataset(arquivo)


def test_nomes_base_colidindo_sao_rejeitados(dataset_tecnico):
    arquivo, adicionar = dataset_tecnico
    adicionar("train", nome="pasta_a/mesmo", valor=10)
    adicionar("train", nome="pasta_b/mesmo", valor=20)
    with pytest.raises(ErroDataset, match="nomes-base colidem"):
        validar_dataset(arquivo, ("train",))


def test_manifesto_e_yaml_normalizado_preservam_origem(dataset_tecnico, tmp_path):
    arquivo, adicionar = dataset_tecnico
    adicionar("train", nome="subpasta/treino", valor=10)
    adicionar("val", nome="val", valor=20)
    conteudo_original = arquivo.read_bytes()
    validado = validar_dataset(arquivo)
    assert validado.contagens["test"]["imagens"] == 0
    assert len(validado.manifesto) == 2
    assert all(len(item["sha256_imagem"]) == 64 for item in validado.manifesto)
    assert len(validado.sha256_conjunto) == 64
    destino = salvar_yaml_normalizado(validado, tmp_path / "execucao" / "dataset.yaml")
    normalizado = yaml.safe_load(destino.read_text(encoding="utf-8"))
    assert Path(normalizado["train"]).is_absolute()
    assert Path(normalizado["train"]) == arquivo.parent / "images/train"
    assert "download" not in normalizado
    assert arquivo.read_bytes() == conteudo_original
    with pytest.raises(FileExistsError):
        salvar_yaml_normalizado(validado, destino)
