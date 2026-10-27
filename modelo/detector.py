"""Adaptador YOLO com pesos locais. Nenhum modelo é baixado pelo aplicativo."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
from time import perf_counter

import cv2

from processamento.preprocessamento import Imagem, preparar_para_modelo

CLASSES_MVP = {0: "plastico"}


class ErroModelo(RuntimeError):
    """Falha compreensível de carregamento ou inferência."""


def sha256_arquivo(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def verificar_pesos(caminho: str | Path) -> Path:
    """Exige um arquivo .pt local não vazio antes de importar Ultralytics."""
    pesos = Path(caminho).expanduser().resolve()
    if not pesos.is_file():
        raise ErroModelo(f"Modelo não encontrado: {pesos.name}. Inclua pesos treinados em modelos/.")
    if pesos.suffix.lower() != ".pt" or pesos.stat().st_size == 0:
        raise ErroModelo("Os pesos devem ser um arquivo .pt válido e não vazio.")
    return pesos


def carregar_yolo_local(caminho: str | Path):
    """Carrega pesos fornecidos localmente; adequado para treino ou inferência.

    Use apenas pesos de origem confiável: checkpoints PyTorch contêm objetos
    serializados. Não há fallback para COCO ou instalação automática de pacotes.
    """
    pesos = verificar_pesos(caminho)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    try:
        from ultralytics import YOLO
        from ultralytics.utils.checks import check_yolov5u_filename
    except (ImportError, OSError) as exc:
        raise ErroModelo("Ultralytics/PyTorch indisponível. Instale requirements.txt e verifique "
                         "a instalação do PyTorch para seu sistema.") from exc
    try:
        normalizado = check_yolov5u_filename(str(pesos).strip().replace("'", ""), verbose=False)
        if normalizado != str(pesos):
            raise ErroModelo("O Ultralytics alteraria o nome deste caminho. Use um caminho sem "
                             "apóstrofos e um nome personalizado como best.pt para evitar busca de outros pesos.")
        modelo = YOLO(str(pesos), task="detect")
        if modelo.task != "detect":
            raise ErroModelo("Use pesos de detecção de objetos, e não classificação ou segmentação.")
        return modelo
    except ErroModelo:
        raise
    except Exception as exc:
        raise ErroModelo(f"Falha ao carregar {pesos.name}. Verifique a integridade e compatibilidade dos pesos.") from exc


@dataclass(frozen=True)
class Deteccao:
    classe_id: int
    classe: str
    confianca: float
    caixa_xyxy: tuple[float, float, float, float]

    def como_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResultadoDeteccao:
    imagem_bgr: Imagem
    deteccoes: list[Deteccao]
    metadados: dict


def desenhar_deteccoes(imagem: Imagem, deteccoes: list[Deteccao]) -> Imagem:
    """Desenha em uma cópia BGR, sem depender de fontes externas."""
    saida = imagem.copy()
    altura, largura = saida.shape[:2]
    for item in deteccoes:
        x1, y1, x2, y2 = (int(round(v)) for v in item.caixa_xyxy)
        x1, x2 = min(x1, largura - 1), min(x2, largura - 1)
        y1, y2 = min(y1, altura - 1), min(y2, altura - 1)
        cv2.rectangle(saida, (x1, y1), (x2, y2), (40, 200, 60), 2)
        texto = f"{item.classe.upper()} {item.confianca:.1%}"
        cv2.putText(saida, texto, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(saida, texto, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (40, 230, 80), 1, cv2.LINE_AA)
    return saida


class DetectorPlastico:
    """Detector desacoplado da interface. Classes adicionais exigem modelo correspondente."""

    def __init__(self, caminho_pesos: str | Path, classes: dict[int, str] | None = None):
        self.caminho_pesos = verificar_pesos(caminho_pesos)
        self.modelo = carregar_yolo_local(self.caminho_pesos)
        self.classes = dict(CLASSES_MVP if classes is None else classes)
        try:
            nomes = self.modelo.names
            nomes = dict(enumerate(nomes)) if isinstance(nomes, list) else dict(nomes)
            normalizados = {int(i): str(nome).strip().lower() for i, nome in nomes.items()}
        except Exception as exc:
            raise ErroModelo("O arquivo de pesos não contém metadados de classes válidos.") from exc
        if normalizados != self.classes:
            raise ErroModelo("Classes incompatíveis. O MVP exige somente a classe 0: plastico. "
                             "Pesos genéricos (por exemplo COCO) não detectam a classe deste projeto.")
        self.sha256_pesos = sha256_arquivo(self.caminho_pesos)

    def inferir(self, imagem: Imagem, *, confianca: float = 0.25, iou: float = 0.7,
                imgsz: int = 640, dispositivo: str = "cpu") -> ResultadoDeteccao:
        """Retorna caixas em pixels da imagem original e apenas escores reais do YOLO."""
        if not 0 < confianca <= 1 or not 0 < iou <= 1:
            raise ValueError("Confiança e IoU devem estar entre 0 (exclusivo) e 1.")
        if not isinstance(imgsz, int) or imgsz < 32 or imgsz % 32:
            raise ValueError("imgsz deve ser um múltiplo de 32, maior ou igual a 32.")
        entrada = preparar_para_modelo(imagem)
        inicio = perf_counter()
        try:
            resultados = self.modelo.predict(source=entrada, conf=confianca, iou=iou,
                                              imgsz=imgsz, device=dispositivo,
                                              save=False, verbose=False)
            if len(resultados) != 1 or resultados[0].boxes is None:
                raise ErroModelo("O modelo não retornou um resultado de detecção válido.")
            caixas = resultados[0].boxes.cpu().numpy()
            deteccoes = []
            altura, largura = entrada.shape[:2]
            for caixa, score, classe in zip(caixas.xyxy, caixas.conf, caixas.cls):
                valores = [float(v) for v in caixa]
                score, classe = float(score), float(classe)
                if (not all(math.isfinite(v) for v in [*valores, score, classe])
                        or not 0 <= score <= 1 or not classe.is_integer()
                        or int(classe) not in self.classes):
                    raise ErroModelo("O modelo retornou uma caixa, confiança ou classe inválida.")
                x1, y1, x2, y2 = valores
                x1, x2 = max(0.0, min(x1, largura)), max(0.0, min(x2, largura))
                y1, y2 = max(0.0, min(y1, altura)), max(0.0, min(y2, altura))
                if x2 <= x1 or y2 <= y1:
                    raise ErroModelo("O modelo retornou uma caixa sem área válida.")
                deteccoes.append(Deteccao(int(classe), self.classes[int(classe)], score, (x1, y1, x2, y2)))
        except ErroModelo:
            raise
        except Exception as exc:
            raise ErroModelo("Falha na inferência. Verifique os pesos, o dispositivo e a memória disponível.") from exc
        metadados = {"data_utc": datetime.now(timezone.utc).isoformat(),
                     "pesos": self.caminho_pesos.name, "sha256_pesos": self.sha256_pesos,
                     "confianca_minima": confianca, "iou": iou, "imgsz": imgsz,
                     "dispositivo": dispositivo, "duracao_segundos": perf_counter() - inicio,
                     "resolucao_original": [largura, altura], "coordenadas": "xyxy em pixels da imagem original",
                     "entrada_modelo": "BGR original; normalização e letterbox internos do YOLO"}
        return ResultadoDeteccao(desenhar_deteccoes(entrada, deteccoes), deteccoes, metadados)
