import cv2
import numpy as np


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def stretch_contrast(gray):
    """Faded ink: spread pixel values so darkest=0 and lightest=255."""
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def denoise(gray):
    """Non-local means removes grain while keeping letter edges."""
    return cv2.fastNlMeansDenoising(gray, None, h=12,
                                    templateWindowSize=7, searchWindowSize=21)


def sharpen(gray):
    """Unsharp mask: subtract a blurred copy to make edges crisper."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def estimate_skew_angle(gray, max_angle=10, step=0.5):
    """
    Try many small rotations on a shrunken copy. When text lines are
    perfectly horizontal, the row sums of black pixels have the biggest
    ups and downs (lines vs gaps). Pick the angle with the biggest variance.
    """
    scale = 0.25
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    center = (w / 2, h / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST)
        score = np.var(rotated.sum(axis=1))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def threshold(gray):
    """Otsu picks the best black/white cutoff automatically."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def crop_to_content(binary, pad=40):
    """Find where the black text is, and cut around it."""
    ys, xs = np.where(binary < 128)
    if len(xs) == 0:
        return binary
    h, w = binary.shape
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h)
    return binary[y0:y1, x0:x1]


def clean_image(img):
    """Full pipeline. Returns (cleaned_image, info_dict)."""
    gray = to_gray(img)
    gray = stretch_contrast(gray)
    gray = denoise(gray)
    gray = sharpen(gray)
    angle = estimate_skew_angle(gray)
    gray = rotate(gray, -angle)
    binary = threshold(gray)
    binary = crop_to_content(binary)
    blur_score = float(cv2.Laplacian(to_gray(img), cv2.CV_64F).var())
    return binary, {"skew_angle": angle, "blur_score": blur_score}