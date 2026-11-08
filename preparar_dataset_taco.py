"""Prepara os dados fornecidos em APS.zip ou dataset_plasticos.zip, sem alterar a origem."""
import argparse
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile, BadZipFile

from treinamento.preparacao import extrair_fotos, reconstruir_labels


def preparar(arquivo: Path, saida: Path, anotacoes: Path) -> Path:
    """Lê ZIP de fotos ou o ZIP externo APS, sempre para uma pasta nova."""
    if not arquivo.is_file():
        raise ValueError(f"Arquivo não encontrado: {arquivo}")
    if saida.exists():
        raise ValueError("A pasta de saída já existe. Escolha outra para preservar seus dados.")
    with ZipFile(arquivo) as pacote:
        internos = [e for e in pacote.infolist() if e.filename.endswith("/dataset_plasticos.zip")]
        if internos:
            if len(internos) != 1:
                raise ValueError("Há mais de um dataset_plasticos.zip no arquivo.")
            saida.parent.mkdir(parents=True, exist_ok=True)
            # O ZIP interno é grande; temporário em disco, não em memória.
            with tempfile.TemporaryDirectory(prefix="aps-extracao-", dir=saida.parent) as pasta:
                interno = Path(pasta) / "dataset_plasticos.zip"
                with pacote.open(internos[0]) as origem, interno.open("wb") as destino:
                    shutil.copyfileobj(origem, destino, length=1024 * 1024)
                extrair_fotos(interno, saida)
        else:
            extrair_fotos(arquivo, saida)
    relatorio = reconstruir_labels(saida, anotacoes)
    print(f"Conversão concluída: {relatorio['caixas']} caixas provenientes do COCO original.")
    print(f"Caixas ajustadas aos limites: {relatorio['caixas_interseccionadas_com_a_imagem']}")
    print("Revise as categorias e as caixas visualmente antes de treinar.")
    return saida / "plastico.yaml"


def main() -> int:
    raiz = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arquivo", type=Path, required=True)
    parser.add_argument("--saida", type=Path, default=raiz / "dados_treinamento")
    parser.add_argument("--anotacoes", type=Path, default=raiz / "dataset/taco/annotations.json")
    args = parser.parse_args()
    try:
        yaml = preparar(args.arquivo.resolve(), args.saida.resolve(), args.anotacoes.resolve())
    except (ValueError, OSError, BadZipFile) as exc:
        print(f"Preparação interrompida: {exc}")
        return 1
    print(f"Configuração gerada: {yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
