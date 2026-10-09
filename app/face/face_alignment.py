import cv2
import numpy as np


# YuNet landmark order:
# right eye, left eye, nose, right mouth corner, left mouth corner
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(
    image: np.ndarray,
    face_row: np.ndarray,
    output_size: int = 112,
) -> np.ndarray | None:
    """Align a YuNet-detected face to the standard 112x112 face template."""

    if face_row is None or len(face_row) < 14:
        return None

    landmarks = np.array([
        [face_row[4], face_row[5]],
        [face_row[6], face_row[7]],
        [face_row[8], face_row[9]],
        [face_row[10], face_row[11]],
        [face_row[12], face_row[13]],
    ], dtype=np.float32)

    if not np.isfinite(landmarks).all():
        return None

    scale = output_size / 112.0
    dst = _ARCFACE_DST * scale

    transform, _ = cv2.estimateAffinePartial2D(
        landmarks,
        dst,
        method=cv2.LMEDS,
    )

    if transform is None:
        return None

    aligned = cv2.warpAffine(
        image,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return aligned
