"""Fine-tuning real com dataset anotado e pesos pré-treinados locais.

Execução: python -m modelo.treinamento --pesos-base caminho/yolo11n.pt
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

from modelo.dados import (RAIZ_PROJETO, YAML_PADRAO, ErroDataset, novo_diretorio,
                          salvar_json_exclusivo, salvar_yaml_normalizado,
                          validar_dataset, versoes_ambiente)


def validar_parametros(*, imgsz: int, batch: int, seed: int) -> None:
    """Mantém os parâmetros básicos explícitos e reprodutíveis."""
    if type(imgsz) is not int or imgsz < 32 or imgsz % 32:
        raise ValueError("imgsz deve ser um múltiplo de 32, maior ou igual a 32.")
    if type(batch) is not int or batch <= 0:
        raise ValueError("batch deve ser um inteiro positivo; reduza-o se faltar memória.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed deve ser um inteiro entre 0 e 4294967295.")


def treinar(pesos_base: str | Path, *, dataset: str | Path = YAML_PADRAO,
            epochs: int = 50, imgsz: int = 640, batch: int = 8,
            device: str = "cpu", seed: int = 42,
            pasta_execucoes: str | Path = RAIZ_PROJETO / "resultados" / "treinamentos",
            pasta_modelos: str | Path = RAIZ_PROJETO / "modelos") -> Path:
    """Treina somente após validar dados e retorna uma cópia exclusiva de best.pt.

    Não cria exemplos, labels ou pesos de demonstração. O modelo base deve
    ter sido obtido conscientemente pelo responsável pelo treinamento.
    """
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs deve ser um inteiro positivo.")
    validar_parametros(imgsz=imgsz, batch=batch, seed=seed)
    dados = validar_dataset(dataset, ("train", "val"))
    # Dataset vazio é rejeitado antes de importar ou instanciar YOLO.
    from modelo.detector import (DetectorPlastico, ErroModelo, carregar_yolo_local,
                                 sha256_arquivo, verificar_pesos)

    base = verificar_pesos(pesos_base)
    hash_base = sha256_arquivo(base)
    modelo = carregar_yolo_local(base)
    execucao = novo_diretorio(Path(pasta_execucoes).resolve(), "treino")
    yaml_runtime = salvar_yaml_normalizado(dados, execucao / "dataset.runtime.yaml")
    parametros = {"epochs": epochs, "imgsz": imgsz, "batch": batch,
                  "device": device, "seed": seed, "workers": 0,
                  "amp": False, "plots": False, "deterministic": True,
                  "cache": False, "pretrained": True}
    inicio = datetime.now(timezone.utc).isoformat()
    try:
        modelo.train(data=str(yaml_runtime), **parametros,
                     project=str(execucao), name="yolo", exist_ok=False, save=True)
        melhor = verificar_pesos(modelo.trainer.best)
        # Confirma que o checkpoint final tem as classes do dataset utilizado.
        DetectorPlastico(melhor, classes=dados.configuracao.classes)
    except (ImportError, OSError) as exc:
        raise ErroModelo("Treinamento interrompido: verifique bibliotecas, espaço em disco "
                         "e permissões de escrita. Consulte a causa no ambiente local.") from exc
    except ErroModelo:
        raise
    except Exception as exc:
        raise ErroModelo("Treinamento não concluído. Verifique pesos, dataset, dispositivo "
                         "e memória; nenhum modelo foi promovido para uso na aplicação. "
                         f"Detalhe: {exc}") from exc
    destino_modelos = Path(pasta_modelos).resolve()
    destino_modelos.mkdir(parents=True, exist_ok=True)
    destino = destino_modelos / f"plastico_{execucao.name}.pt"
    criado = False
    try:
        with destino.open("xb") as saida:
            criado = True
            with melhor.open("rb") as origem:
                shutil.copyfileobj(origem, saida)
    except OSError:
        # Evita que uma cópia parcial apareça como candidato no aplicativo.
        if criado and destino.exists():
            destino.unlink()
        raise
    relatorio = {
        "tipo": "treinamento_concluido", "inicio_utc": inicio,
        "fim_utc": datetime.now(timezone.utc).isoformat(),
        "pesos_base": str(base), "sha256_pesos_base": hash_base,
        "melhor_checkpoint": str(melhor), "modelo_copiado": str(destino),
        "sha256_modelo": sha256_arquivo(destino),
        "parametros": parametros, "dataset": dados.proveniencia(),
        "yaml_runtime": str(yaml_runtime), "versoes": versoes_ambiente(),
        "observacao": "Treinamento concluído não comprova generalização. Avalie o split test separado.",
    }
    salvar_json_exclusivo(execucao / "treinamento.json", relatorio)
    return destino


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tuning YOLO com fotografias e anotações reais.")
    parser.add_argument("--dataset", type=Path, default=YAML_PADRAO, help="YAML local do dataset.")
    parser.add_argument("--pesos-base", type=Path, required=True,
                        help="Arquivo .pt pré-treinado local, obtido explicitamente (ex.: yolo11n.pt).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu ou índice da GPU, por exemplo 0.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    try:
        melhor = treinar(**vars(args))
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Dependência ausente ({exc.name}). Execute python -m pip install -r requirements.txt.",
              file=sys.stderr)
        return 1
    except (ErroDataset, ValueError, RuntimeError, OSError) as exc:
        print(f"Treinamento não executado ou interrompido: {exc}", file=sys.stderr)
        return 1
    print(f"Melhor modelo salvo em: {melhor}")
    print("Avalie-o em um conjunto de teste independente antes de apresentar desempenho.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
