import os

# Must be set before importing PaddleOCR / paddle.
os.environ["FLAGS_use_mkldnn"] = "False"
os.environ["FLAGS_enable_mkldnn"] = "False"
os.environ["FLAGS_enable_pir_api"] = "False"
os.environ["FLAGS_use_onednn"] = "False"

from paddleocr import PaddleOCR


class PrintedOCR:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            enable_mkldnn=False,
            show_log=False,
        )

    def extract_text(self, image_path: str) -> dict:
        result = self.ocr.ocr(image_path, cls=True)

        lines = []
        boxes = []
        confidences = []

        if not result or not result[0]:
            return {
                "text": "",
                "confidence": 0.0,
                "boxes": [],
            }

        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            confidence = float(line[1][1])

            lines.append(text)
            confidences.append(confidence)
            boxes.append(
                {
                    "bbox": bbox,
                    "text": text,
                    "confidence": confidence,
                }
            )

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "text": "\n".join(lines),
            "confidence": avg_confidence,
            "boxes": boxes,
        }