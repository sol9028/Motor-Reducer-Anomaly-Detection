# 출처: 보고서/프로젝트결과보고서_클로드.docx 2.6~3.1절 본문·표·그림 1
import sys, os
sys.path.insert(0, os.path.expanduser("~/.claude/skills/slides"))
import slidechart as sc
from PIL import Image
import zipfile, io

A = "slides/assets/"

# 학습 시간(9슬롯): 1일 4시간 26분 = 1,706분 / 1시간 52분 = 112분 / 1시간 6분 = 66분
fig, ax = sc.figure("col")
sc.bar(ax, ["기준 모델", "속도 개선", "12채널 융합"], [1706, 112, 66],
       emphasize=[1], fmt="{:,.0f}분", horizontal=True)
sc.save(fig, A + "train_time.png")

# 차종별 macro-F1: 센서별(late) vs 12채널 융합(early)
fig, ax = sc.figure("col")
labels = ["IONIQ\n센서별", "IONIQ\n융합", "KONA\n센서별", "KONA\n융합", "NIRO\n센서별", "NIRO\n융합"]
sc.bar(ax, labels, [0.657, 0.871, 0.673, 0.874, 0.628, 0.826], emphasize=[1, 3, 5], fmt="{:.3f}")
sc.save(fig, A + "f1_fusion.png")

# 보고서 그림 1(차종 3행 × 센서 4열)에서 IONIQ 행만 / IONIQ의 Vib_TM·12채널 융합만 잘라 냄
with zipfile.ZipFile("보고서/프로젝트결과보고서_클로드.docx") as z:
    im = Image.open(io.BytesIO(z.read("word/media/image1.png")))
w, h = im.size
im.crop((0, 0, w, int(h * 0.335))).save(A + "confusion_ioniq_row.png")
im.crop((int(w * 0.505), 0, w, int(h * 0.335))).save(A + "confusion_ioniq.png")
