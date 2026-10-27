"""Avaliação YOLO com métricas reais e conjunto explicitamente selecionado.

Execução: python -m modelo.avaliacao --pesos modelos/seu_modelo.pt --split test
"""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

from modelo.dados import (RAIZ_PROJETO, YAML_PADRAO, ErroDataset, novo_diretorio,
                          salvar_json_exclusivo, salvar_yaml_normalizado,
                          validar_dataset, versoes_ambiente)
from modelo.treinamento import validar_parametros


def extrair_metricas(resultado) -> dict[str, float]:
    """Obtém quatro métricas de DetMetrics sem preencher ausências com zeros."""
    try:
        metricas = {"precision": float(resultado.box.mp), "recall": float(resultado.box.mr),
                    "mAP50": float(resultado.box.map50), "mAP50-95": float(resultado.box.map)}
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("YOLO não retornou as quatro métricas de detecção esperadas. "
                           "Nenhum arquivo de métricas foi gravado.") from exc
    if any(not math.isfinite(valor) or not 0 <= valor <= 1 for valor in metricas.values()):
        raise RuntimeError("YOLO retornou métricas inválidas. Nenhum arquivo de métricas foi gravado.")
    return metricas


def avaliar(pesos: str | Path, *, dataset: str | Path = YAML_PADRAO,
            split: str = "test", imgsz: int = 640, batch: int = 8,
            device: str = "cpu", conf: float = 0.001, iou: float = 0.7,
            pasta_metricas: str | Path = RAIZ_PROJETO / "resultados" / "metricas") -> Path:
    """Avalia modelo compatível e salva JSON somente após retorno válido do YOLO.

    Use val para ajustar decisões durante o desenvolvimento. Reserve test
    para a avaliação final, sem escolher parâmetros pelos seus resultados.
    """
    if split not in {"val", "test"}:
        raise ValueError("A avaliação deve usar --split val ou --split test.")
    validar_parametros(imgsz=imgsz, batch=batch, seed=0)
    if not 0 < conf <= 1 or not 0 < iou <= 1:
        raise ValueError("conf e iou devem estar entre 0 (exclusivo) e 1.")
    dados = validar_dataset(dataset, (split,))
    from modelo.detector import DetectorPlastico, ErroModelo

    detector = DetectorPlastico(pesos, classes=dados.configuracao.classes)
    execucao = novo_diretorio(Path(pasta_metricas).resolve(), f"avaliacao_{split}")
    yaml_runtime = salvar_yaml_normalizado(dados, execucao / "dataset.runtime.yaml")
    parametros = {"split": split, "imgsz": imgsz, "batch": batch, "device": device,
                  "conf": conf, "iou": iou, "workers": 0, "plots": False,
                  "save_json": False, "save_txt": False, "half": False}
    inicio = datetime.now(timezone.utc).isoformat()
    try:
        resultado = detector.modelo.val(data=str(yaml_runtime), **parametros,
                                         project=str(execucao), name="yolo", exist_ok=False)
        metricas = extrair_metricas(resultado)
    except (ImportError, OSError) as exc:
        raise ErroModelo("Avaliação interrompida: verifique dependências e permissões. "
                         "Nenhum JSON de métricas foi gravado.") from exc
    except Exception as exc:
        raise ErroModelo("Avaliação não concluída. Verifique pesos, dispositivo e dataset. "
                         f"Nenhum JSON de métricas foi gravado. Detalhe: {exc}") from exc
    relatorio = {
        "tipo": "avaliacao_real_yolo", "inicio_utc": inicio,
        "fim_utc": datetime.now(timezone.utc).isoformat(),
        "pesos": str(detector.caminho_pesos), "sha256_pesos": detector.sha256_pesos,
        "split": split, "metricas": metricas, "escala_metricas": "0 a 1",
        "parametros": parametros, "dataset": dados.proveniencia(),
        "yaml_runtime": str(yaml_runtime), "versoes": versoes_ambiente(),
        "interpretacao": {
            "precision_recall": "Resumos agregados fornecidos pelo YOLO; não são confiança de uma caixa.",
            "mAP50": "AP média com limiar de IoU 0,50.",
            "mAP50-95": "AP média nos limiares de IoU 0,50 a 0,95, passo 0,05.",
            "limite": "Válido para o conjunto e os parâmetros registrados; não comprova desempenho em outros ambientes.",
        },
    }
    caminho = execucao / "metricas.json"
    salvar_json_exclusivo(caminho, relatorio)
    return caminho


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avaliação real do modelo de resíduos plásticos.")
    parser.add_argument("--dataset", type=Path, default=YAML_PADRAO)
    parser.add_argument("--pesos", type=Path, required=True, help="Arquivo .pt treinado para as classes do YAML.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Limiar baixo para construir curvas de precisão/recall (padrão YOLO).")
    parser.add_argument("--iou", type=float, default=0.7, help="Limiar de IoU do NMS; não o limiar da mAP.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    try:
        caminho = avaliar(**vars(args))
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"Dependência ausente ({exc.name}). Execute python -m pip install -r requirements.txt.",
              file=sys.stderr)
        return 1
    except (ErroDataset, ValueError, RuntimeError, OSError) as exc:
        print(f"Avaliação não executada ou interrompida: {exc}", file=sys.stderr)
        return 1
    print(f"Métricas reais e proveniência salvas em: {caminho}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
