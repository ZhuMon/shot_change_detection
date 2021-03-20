import sys
from PIL import Image
import numpy as np

PAIR_WISE_t = 50
PAIR_WISE_T = 42  # persent
TWIN_COMPARISON_Tb = 60
TWIN_COMPARISON_Ts = 10
TWIN_COMPARISON_accum = 0
GRADUAL_TRANSITION_FRAME = []


def check_fade_or_wipe(SD, index):
    global GRADUAL_TRANSITION_FRAME
    global TWIN_COMPARISON_accum
    if SD > TWIN_COMPARISON_Tb:
        if len(GRADUAL_TRANSITION_FRAME) != 0:
            if TWIN_COMPARISON_accum > TWIN_COMPARISON_Tb:
                print(
                    f"{GRADUAL_TRANSITION_FRAME[0]}~{GRADUAL_TRANSITION_FRAME[-1]}")
            GRADUAL_TRANSITION_FRAME = []
        TWIN_COMPARISON_accum = TWIN_COMPARISON_Ts
        return True
    elif SD > TWIN_COMPARISON_Ts:
        TWIN_COMPARISON_accum += SD - TWIN_COMPARISON_Ts
        GRADUAL_TRANSITION_FRAME.append(index)
        return False
    else:
        if len(GRADUAL_TRANSITION_FRAME) != 0:
            if TWIN_COMPARISON_accum > TWIN_COMPARISON_Tb:
                print(
                    f"{GRADUAL_TRANSITION_FRAME[0]}~{GRADUAL_TRANSITION_FRAME[-1]}")
            GRADUAL_TRANSITION_FRAME = []
        TWIN_COMPARISON_accum = TWIN_COMPARISON_Ts
        return False


def pair_wise(first, second, _):
    im1 = Image.open(first)
    im2 = Image.open(second)

    assert im1.size == im2.size, "size of two image not the same"
    pix1 = im1.load()
    pix2 = im2.load()

    DP = 0
    # compare pixel in two frame at same position
    for i in range(im1.size[0]):
        for j in range(im1.size[1]):
            # compare every intensity, and sum them
            d = 0
            for k in range(3):
                d += abs(pix2[i, j][k] - pix1[i, j][k])
            if d > PAIR_WISE_t:
                DP += 1

    # print(DP/(im1.size[0]*im1.size[1]) * 100, DP)
    # Normalize
    if DP/(im1.size[0]*im1.size[1]) * 100 > PAIR_WISE_T:
        return True
    else:
        return False


def histogram_comparison(first, second, index):
    # only grey level
    im1 = Image.open(first).convert('LA')
    im2 = Image.open(second).convert('LA')

    assert im1.size == im2.size, "size of two image not the same"
    pix1 = im1.load()
    pix2 = im2.load()

    his1 = dict.fromkeys(range(0, 256), 0)
    his2 = dict.fromkeys(range(0, 256), 0)
    for i in range(im1.size[0]):
        for j in range(im1.size[1]):
            his1[pix1[i, j][0]] += 1
            his2[pix2[i, j][0]] += 1

    SD = 0
    for i in range(256):
        SD += abs(his1[i] - his2[i])

    # Normalize
    SD_n = SD/(im1.size[0]*im1.size[1]) * 100
    return check_fade_or_wipe(SD_n, index)


def color_histogram_comp(first, second, index):
    im1 = Image.open(first)
    im2 = Image.open(second)

    assert im1.size == im2.size, "size of two image not the same"
    pix1 = im1.load()
    pix2 = im2.load()
his1 = {} his2 = {}
    for i in range(im1.size[0]):
        for j in range(im1.size[1]):
            his1[pix1[i, j]] = 0 if pix1[i,
                                         j] not in his1.keys() else his1[pix1[i, j]] + 1
            his2[pix2[i, j]] = 0 if pix2[i,
                                         j] not in his2.keys() else his2[pix2[i, j]] + 1

    color_union = set(his1.keys()).union(his2.keys())

    CHD = 0
    for c in color_union:
        if c not in his2:
            CHD += his1[c]
        elif c not in his1:
            CHD += his2[c]
        else:
            CHD += abs(his1[c] - his2[c])

    CHD_n = CHD / (im1.size[0] * im1.size[1]) * 100
    # print(CHD_n)
    # return False
    return check_fade_or_wipe(CHD_n, index)

def likelihood_ratio(first, second, index):
    im1 = Image.open(first).convert('LA')
    im2 = Image.open(second).convert('LA')

    assert im1.size == im2.size, "size of two image not the same"
    pix1 = im1.load()
    pix2 = im2.load()

    sum1 = 0
    sum2 = 0
    expanded_pix1 = []
    expanded_pix2 = []
    for i in range(im1.size[0]):
        for j in range(im1.size[1]):
            sum1 += pix1[i,j][0]
            sum2 += pix2[i,j][0]
            expanded_pix1.append(pix1[i,j][0])
            expanded_pix2.append(pix2[i,j][0])

    mean1 = sum1/(im1.size[0]*im1.size[1])
    mean2 = sum2/(im1.size[0]*im1.size[1])

    cov1 = np.cov(expanded_pix1)
    cov2 = np.cov(expanded_pix2)

    LR = (((cov1 + cov2)/2 + ((mean1-mean2)/2)**2)**2) / 2 / (im1.size[0]*im1.size[1]) *100
    print(LR)
    return check_fade_or_wipe(LR, index)

def answer(n="news"):
    with open(f"{n}_ground.txt", 'r') as f:
        shot_change = []
        for l in f.readlines()[4:]:
            if l.find('~') >= 0:
                r = l.split('~')
                # add previous frame to correct two frame comparison
                shot_change.append(int(r[0])-1) 
                for rr in range(int(r[0]), int(r[1].replace('\n', ''))+1):
                    shot_change.append(rr-1)
            else:
                shot_change.append(int(l.replace('\n', ''))-1)

    return shot_change


def read_news(compare=pair_wise):
    # T = 42
    for i in range(0, 1379):
        # for i in answer("news"):
        if compare(f"news_out/news-000{i:04}.jpg", f"news_out/news-000{i+1:04}.jpg"):
            print(i+1)


def read_soccer(compare=pair_wise):
    for i in range(0, 864):
    # for i in answer("soccer"):
        if compare(f"soccer_out/soccer{i:07}.jpg", f"soccer_out/soccer{i+1:07}.jpg", i):
            pass
            # print(i+1)


def read_ngc(compare=pair_wise):
    for i in range(0, 1059):
        # for i in answer("ngc"):
        if i % 100 == 0:
            print(f"{i:04}/1059")
        if compare(f"ngc_out/ngc-{i:04}.jpeg", f"ngc_out/ngc-{i+1:04}.jpeg", i):
            print(i+1)


if __name__ == "__main__":
    # read_news()
    # print(answer("news"))
    # print(answer("ngc"))
    # read_ngc()
    # read_soccer(color_histogram_comp)
    # read_soccer(likelihood_ratio)
    # read_soccer(histogram_comparison)
