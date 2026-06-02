from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import cv2
import numpy as np
import fitz  # PyMuPDF
from typing import List, Dict, Any


app = FastAPI(title="Archive OpenCV Service", version="1.0.0")


@app.get("/health")
def health():
    return {
        "success": True,
        "service": "archive-opencv-service",
        "opencv_version": cv2.__version__
    }


def read_image_from_bytes(file_bytes: bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def image_to_png_bytes(image) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("图片编码失败")
    return buffer.tobytes()


def get_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def calc_blur_score(image) -> float:
    """
    模糊检测：
    分数越低越模糊。
    常见经验：
    < 60：很模糊
    60-100：可能模糊
    > 100：相对清晰
    具体阈值要根据你的扫描件样本调。
    """
    gray = get_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_blank_page(image) -> Dict[str, Any]:
    """
    空白页检测：
    根据非白色像素占比判断。
    """
    gray = get_gray(image)

    # 大于 245 认为是白色
    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    total_pixels = binary.size
    white_pixels = int(np.sum(binary == 255))
    white_ratio = white_pixels / total_pixels

    # 非白区域比例
    content_ratio = 1 - white_ratio

    return {
        "white_ratio": round(float(white_ratio), 6),
        "content_ratio": round(float(content_ratio), 6),
        "is_blank": content_ratio < 0.002
    }


def detect_black_border(image) -> Dict[str, Any]:
    """
    黑边检测：
    检查页面四周区域是否存在明显黑色像素。
    """
    gray = get_gray(image)
    h, w = gray.shape

    border_size_h = max(5, int(h * 0.03))
    border_size_w = max(5, int(w * 0.03))

    top = gray[:border_size_h, :]
    bottom = gray[h - border_size_h:, :]
    left = gray[:, :border_size_w]
    right = gray[:, w - border_size_w:]

    def black_ratio(area):
        return float(np.sum(area < 40) / area.size)

    result = {
        "top_black_ratio": round(black_ratio(top), 6),
        "bottom_black_ratio": round(black_ratio(bottom), 6),
        "left_black_ratio": round(black_ratio(left), 6),
        "right_black_ratio": round(black_ratio(right), 6)
    }

    result["has_black_border"] = any(v > 0.05 for v in result.values())

    return result


def detect_skew_angle(image) -> Dict[str, Any]:
    """
    歪斜角度检测：
    优先使用页面中的长横线估计小角度倾斜；如果横线不足，再回退到
    minAreaRect，并把 OpenCV 4.x 的 80-90 度结果归一到接近 0 度。
    """
    gray = get_gray(image)
    h, w = gray.shape

    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=120,
        minLineLength=max(80, int(w * 0.25)),
        maxLineGap=20
    )

    horizontal_angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue

            line_angle = float(np.degrees(np.arctan2(dy, dx)))
            if line_angle >= 90:
                line_angle -= 180
            if line_angle < -90:
                line_angle += 180

            if abs(line_angle) <= 15:
                horizontal_angles.append(line_angle)

    if len(horizontal_angles) >= 5:
        angle = -float(np.median(horizontal_angles))
        rotate_angle = -angle

        return {
            "skew_angle": round(float(angle), 3),
            "rotate_angle": round(float(rotate_angle), 3),
            "is_skewed": abs(angle) > 1.0,
            "angle_source": "hough_horizontal"
        }

    # 反色二值化，让文字/线条变成白色
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 100:
        return {
            "skew_angle": 0.0,
            "rotate_angle": 0.0,
            "is_skewed": False,
            "angle_source": "insufficient_content",
            "message": "有效内容过少，无法判断倾斜"
        }

    raw_angle = cv2.minAreaRect(coords)[-1]

    # OpenCV 4.x may report near-vertical boxes as 80-90 degrees.
    # Normalize the result to the small page skew around 0 degrees.
    if raw_angle < -45:
        angle = -(90 + raw_angle)
    elif raw_angle <= 0:
        angle = -raw_angle
    elif raw_angle > 45:
        angle = raw_angle - 90
    else:
        angle = raw_angle

    rotate_angle = -angle

    return {
        "skew_angle": round(float(angle), 3),
        "rotate_angle": round(float(rotate_angle), 3),
        "is_skewed": abs(angle) > 1.0,
        "angle_source": "min_area_rect"
    }


def rotate_image(image, angle: float):
    """
    按角度旋转图片，保持原尺寸。
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def preprocess_image(image):
    """
    基础预处理：
    1. 灰度化
    2. 去噪
    3. 自适应二值化

    注意：不是所有 OCR 都适合二值图。
    MinerU / PaddleOCR 有时直接吃原图效果更好。
    所以建议保留原图，同时保存预处理图用于对比。
    """
    gray = get_gray(image)

    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    return binary


def analyze_image(image) -> Dict[str, Any]:
    h, w = image.shape[:2]

    blur_score = calc_blur_score(image)
    blank_result = detect_blank_page(image)
    border_result = detect_black_border(image)
    skew_result = detect_skew_angle(image)

    is_blurry = blur_score < 80

    need_manual_review = (
        is_blurry
        or blank_result["is_blank"]
        or border_result["has_black_border"]
        or skew_result["is_skewed"]
    )

    return {
        "width": w,
        "height": h,
        "blur_score": round(float(blur_score), 3),
        "is_blurry": is_blurry,
        "blank": blank_result,
        "black_border": border_result,
        "skew": skew_result,
        "need_manual_review": need_manual_review
    }


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[np.ndarray]:
    """
    PDF 转图片。
    dpi 越高越清晰，但越慢、越占内存。
    档案扫描件建议 200 或 300。
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        image = img_data.reshape(pix.height, pix.width, pix.n)

        # PyMuPDF 输出 RGB，OpenCV 使用 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        images.append(image)

    doc.close()
    return images


@app.post("/analyze-image")
async def analyze_image_api(file: UploadFile = File(...)):
    content = await file.read()
    image = read_image_from_bytes(content)

    if image is None:
        return {
            "success": False,
            "message": "图片读取失败，请确认上传的是 jpg/png/jpeg 图片"
        }

    result = analyze_image(image)

    return {
        "success": True,
        "filename": file.filename,
        "result": result
    }


@app.post("/preprocess-image")
async def preprocess_image_api(file: UploadFile = File(...)):
    content = await file.read()
    image = read_image_from_bytes(content)

    if image is None:
        return {
            "success": False,
            "message": "图片读取失败"
        }

    skew_result = detect_skew_angle(image)

    # 自动转正
    if skew_result.get("is_skewed"):
        image = rotate_image(image, -skew_result["skew_angle"])

    processed = preprocess_image(image)
    png_bytes = image_to_png_bytes(processed)

    return Response(content=png_bytes, media_type="image/png")


@app.post("/analyze-pdf")
async def analyze_pdf_api(file: UploadFile = File(...), dpi: int = 200):
    content = await file.read()

    try:
        images = pdf_to_images(content, dpi=dpi)
    except Exception as e:
        return {
            "success": False,
            "message": f"PDF解析失败：{str(e)}"
        }

    pages = []

    for index, image in enumerate(images):
        page_no = index + 1
        result = analyze_image(image)

        pages.append({
            "page_index": index,
            "page_no": page_no,
            "suggested_archive_page_code": str(page_no).zfill(3),
            "result": result
        })

    abnormal_pages = [
        page for page in pages
        if page["result"]["need_manual_review"]
    ]

    return {
        "success": True,
        "filename": file.filename,
        "dpi": dpi,
        "total_pages": len(pages),
        "abnormal_page_count": len(abnormal_pages),
        "pages": pages
    }
