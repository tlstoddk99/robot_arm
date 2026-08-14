"""ArUco 마커 4개(ID 0~3) 생성 → 인쇄용 PNG (OpenCV 4.5.x 호환)"""
import cv2
import numpy as np

# 딕셔너리 가져오기 (구형/신형 API 모두 대응)
try:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)      # 구형(4.5.x)
except AttributeError:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # 신형

def gen_marker(d, mid, px):
    # 구형: drawMarker, 신형: generateImageMarker
    if hasattr(cv2.aruco, "drawMarker"):
        return cv2.aruco.drawMarker(d, mid, px)
    return cv2.aruco.generateImageMarker(d, mid, px)

MARKER_PX = 600
for mid in range(4):
    img = gen_marker(aruco_dict, mid, MARKER_PX)
    pad = 80  # 흰 여백 (검출에 필수)
    canvas = np.ones((MARKER_PX + 2*pad, MARKER_PX + 2*pad), np.uint8) * 255
    canvas[pad:pad+MARKER_PX, pad:pad+MARKER_PX] = img
    fname = f"aruco_id{mid}.png"
    cv2.imwrite(fname, canvas)
    print(f"저장: {fname}")

print("\n[인쇄 주의]")
print("- 마커 한 변을 4~5cm로 인쇄 (인쇄 후 자로 실제 크기 측정)")
print("- 흰 여백 자르지 말 것")
print("- 평평하게 부착, 구겨지지 않게")
