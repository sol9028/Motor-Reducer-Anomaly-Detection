"""STFT 설명 장면: 예시 신호 → 창 이동 → 스펙트로그램 → 실제 데이터(NORMAL vs DEMAG).

렌더링 (프로젝트 루트에서):
    .venv/Scripts/manim -qh video/stft_scene.py STFTScene
"""
from pathlib import Path

import numpy as np
from manim import *

FONT = "Malgun Gothic"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "샘플데이터/Sample/01.원천데이터/1.모터_감속기/IONIQ"
NORMAL_PNG = RAW / "NORMAL/0609/Vib_Motor/22_06_09_12_26_53_01_zsplit001_001_Vib_Motor.png"
DEMAG_PNG = RAW / "DEMAG/0728/Vib_Motor/22_07_28_12_15_20_01_zsplit005_001_Vib_Motor.png"

# 예시 신호: 실제 원시 시계열은 데이터셋에 없어서 STFT 원리 설명용으로 합성
FS, N, WIN, HOP, N_ROWS = 1000, 2000, 128, 24, 48


def example_signal():
    t = np.arange(N) / FS
    chirp = np.sin(2 * np.pi * (60 * t + 50 * t**2))  # 60 → 260 Hz로 올라가는 성분
    tone = 0.6 * np.sin(2 * np.pi * 320 * t)          # 일정한 320 Hz 성분
    return t, chirp + tone


def stft_columns(x):
    w = np.hanning(WIN)
    starts = list(range(0, N - WIN + 1, HOP))
    mags = np.array([np.abs(np.fft.rfft(x[s:s + WIN] * w))[:N_ROWS] for s in starts])
    return starts, mags / mags.max()


def jet(v):
    """matplotlib 없이 jet 컬러맵 근사. v: 0~1 배열 → uint8 RGB."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def label(s, size=26, **kw):
    return Text(s, font=FONT, font_size=size, **kw)


class STFTScene(Scene):
    def construct(self):
        t, x = example_signal()
        starts, mags = stft_columns(x)

        title = label("STFT: 진동 신호를 이미지로 바꾸기", 36).to_edge(UP)
        self.play(Write(title))

        # 1) 시간 신호
        ax = Axes(x_range=[0, N / FS, 0.5], y_range=[-2, 2, 1], x_length=11, y_length=2,
                  tips=False, axis_config={"include_ticks": False})
        ax.next_to(title, DOWN, buff=0.6)
        wave = ax.plot_line_graph(t[::2], x[::2], add_vertex_dots=False,
                                  line_color=BLUE_B, stroke_width=1.5)
        sig_label = label("예시 신호 (합성)", 20, color=GRAY_B).next_to(ax, UP, buff=0.05).align_to(ax, LEFT)
        self.play(Create(ax), Create(wave), FadeIn(sig_label), run_time=2)

        # 2) 스펙트로그램 틀
        spec_w, spec_h = 11, 3
        frame = Rectangle(width=spec_w, height=spec_h, stroke_color=GRAY, stroke_width=1)
        frame.next_to(ax, DOWN, buff=0.7)
        freq_lbl = label("주파수 ↑", 20).rotate(PI / 2).next_to(frame, LEFT, buff=0.15)
        time_lbl = label("시간 →", 20).next_to(frame, DOWN, buff=0.15)
        self.play(Create(frame), FadeIn(freq_lbl), FadeIn(time_lbl))

        # 3) 창을 밀면서 열을 하나씩 채우기
        win_w = ax.x_length * WIN / N
        window = Rectangle(width=win_w, height=ax.y_length, stroke_color=YELLOW,
                           fill_color=YELLOW, fill_opacity=0.15)
        window.move_to(ax.c2p(starts[0] / FS, 0), aligned_edge=LEFT)
        note = label("짧은 구간마다 FFT → 한 열", 22, color=YELLOW).next_to(ax, DOWN, buff=0.12)
        self.play(FadeIn(window), FadeIn(note))

        # 열마다 따로 붙이면 경계에 틈이 생겨서, 지금까지의 열을 한 장 이미지로 매번 다시 그림
        full = jet(mags.T[::-1])  # (주파수, 시간, RGB), 저주파가 아래
        col_w = spec_w / len(starts)
        spec = None
        for i, s in enumerate(starts):
            new = ImageMobject(full[:, :i + 1])
            new.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            new.stretch_to_fit_width(col_w * (i + 1)).stretch_to_fit_height(spec_h)
            new.move_to(frame.get_corner(DL), aligned_edge=DL)
            if spec:
                self.remove(spec)
            self.add(new)
            spec = new
            rt = 0.5 if i < 3 else 0.05
            self.play(window.animate.move_to(ax.c2p(s / FS, 0), aligned_edge=LEFT), run_time=rt)
        self.play(FadeOut(window), FadeOut(note))

        point = label("주파수가 오르는 성분은 사선, 일정한 성분은 수평선", 24, color=YELLOW)
        point.move_to(time_lbl)
        self.play(FadeOut(time_lbl), Write(point))
        self.wait(2)

        # 4) 실제 데이터로 전환
        self.play(*[FadeOut(m) for m in self.mobjects if m is not title])
        new_title = label("실제 데이터: IONIQ 모터 진동 (Vib_Motor)", 32).to_edge(UP)
        self.play(Transform(title, new_title))

        imgs = Group()
        for path, name, color in [(NORMAL_PNG, "NORMAL", GREEN), (DEMAG_PNG, "DEMAG (자석 감자)", RED)]:
            img = ImageMobject(str(path)).set_width(6)
            cap = label(name, 26, color=color).next_to(img, DOWN, buff=0.2)
            imgs.add(Group(img, cap))
        imgs.arrange(RIGHT, buff=0.5).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(imgs[0], shift=UP * 0.3))
        self.play(FadeIn(imgs[1], shift=UP * 0.3))

        outro = label("이 이미지를 CNN에 넣어 5개 클래스로 분류", 26).to_edge(DOWN)
        self.play(Write(outro))
        self.wait(3)
