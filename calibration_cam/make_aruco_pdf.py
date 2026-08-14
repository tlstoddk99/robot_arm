"""ArUco 마커 4개를 정확한 물리 크기(4cm)로 PDF 생성"""
import cv2
import numpy as np

# ArUco 마커 생성 (구형 API)
try:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
except AttributeError:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

def gen(mid, px):
    if hasattr(cv2.aruco, "drawMarker"):
        return cv2.aruco.drawMarker(aruco_dict, mid, px)
    return cv2.aruco.generateImageMarker(aruco_dict, mid, px)

# 고해상도로 마커 PNG 생성 (여백 포함)
MARKER_PX = 800
PAD = 100
for mid in range(4):
    m = gen(mid, MARKER_PX)
    canvas = np.ones((MARKER_PX+2*PAD, MARKER_PX+2*PAD), np.uint8) * 255
    canvas[PAD:PAD+MARKER_PX, PAD:PAD+MARKER_PX] = m
    cv2.imwrite(f"/tmp/aruco_{mid}.png", canvas)

# matplotlib으로 정확한 물리 크기 PDF 생성
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

MARKER_CM = 4.0          # 마커 실제 크기
TOTAL_CM = MARKER_CM * (MARKER_PX+2*PAD) / MARKER_PX  # 여백 포함 전체 크기
CM = 1/2.54              # cm → inch

with PdfPages("/home/a/aruco_markers_4cm.pdf") as pdf:
    for mid in range(4):
        img = plt.imread(f"/tmp/aruco_{mid}.png")
        fig = plt.figure(figsize=(TOTAL_CM*CM, TOTAL_CM*CM))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"ID {mid} (marker={MARKER_CM}cm)", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)

print("생성: /home/a/aruco_markers_4cm.pdf (4페이지, 각 마커 4cm)")
print("인쇄 시 반드시 '실제 크기' 또는 '100%' 로 인쇄 (축소/맞춤 끄기)")
