import sys
import argparse

from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt

DRY = False # whether only plot
VIDEO = ""
VALUE_FOR_PLOT = []
FRAME_AMOUNT = 0
FILE_PREFIX = ''
FILE_EXTENSION = ''

PAIR_WISE_t = 50
PAIR_WISE_T = 42  # persent
TWIN_COMPARISON_Tb = 60
TWIN_COMPARISON_Ts = 10
TWIN_COMPARISON_accum = 0
GRADUAL_TRANSITION_FRAME = []
LIKELIHOOD_REGION_SIZE = (8, 8)
EDGE_DETECT_THRESHOLD = 125 # threshold to decide whether edge
ANSWER = []
EXPAND_ANSWER = []

def check_fade_or_wipe(SD, index):
    global GRADUAL_TRANSITION_FRAME
    global TWIN_COMPARISON_accum
    if SD > TWIN_COMPARISON_Tb:
        if len(GRADUAL_TRANSITION_FRAME) != 0:
            if TWIN_COMPARISON_accum > TWIN_COMPARISON_Tb:
                print(
                    f"{GRADUAL_TRANSITION_FRAME[0]}~{GRADUAL_TRANSITION_FRAME[-1]}")
                ANSWER.append(f"{GRADUAL_TRANSITION_FRAME[0]}~{GRADUAL_TRANSITION_FRAME[-1]}")
                EXPAND_ANSWER += GRADUAL_TRANSITION_FRAME
 
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
                ANSWER.append(f"{GRADUAL_TRANSITION_FRAME[0]}~{GRADUAL_TRANSITION_FRAME[-1]}")
            GRADUAL_TRANSITION_FRAME = []
        TWIN_COMPARISON_accum = TWIN_COMPARISON_Ts
        return False


def pair_wise(first, second, index):
    im1 = Image.open(first)
    im2 = Image.open(second)

    assert im1.size == im2.size, "size of two image not the same"
    pix1 = im1.load()
    pix2 = im2.load()

    pix1 = np.array(im1)
    pix2 = np.array(im2)
    DP = 0
    # compare pixel in two frame at same position
    print(pix1[0][1])
    print(pix2[0][1])
    print(abs(pix2 - pix1)[0][1])
    sys.exit(1)
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
    DP_n = DP/(im1.size[0]*im1.size[1]) * 100
    VALUE_FOR_PLOT.append(DP_n)    
    if DRY:
        return False
    return check_fade_or_wipe(DP_n, index)

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
    VALUE_FOR_PLOT.append(SD_n)    
    if DRY:
        return False
    return check_fade_or_wipe(SD_n, index)


def color_histogram_comp(first, second, index):
    im1 = Image.open(first)
    im2 = Image.open(second)

    assert im1.size == im2.size, "size of two image not the same"
    pix1 = im1.load()
    pix2 = im2.load()
    his1 = {}
    his2 = {}
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
    VALUE_FOR_PLOT.append(CHD_n)    
    if DRY:
        return False
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
    VALUE_FOR_PLOT.append(LR)
    if DRY:
        return False
    return check_fade_or_wipe(LR, index)

def edge_detection(first, second, index):
    im1 = Image.open(first).convert('L')
    im2 = Image.open(second).convert('L')

    assert im1.size == im2.size, "size of two image not the same"
    edg1 = im1.filter(ImageFilter.FIND_EDGES)
    edg2 = im2.filter(ImageFilter.FIND_EDGES)

    pix1 = edg1.load()
    pix2 = edg2.load()

    delta_n1 = 0 # number of edge of first image
    delta_n2 = 0
    X_in = 0
    X_out = 0
    for i in range(im1.size[0]):
        for j in range(im1.size[1]):
            # second image has new edge
            if pix2[i,j] >= EDGE_DETECT_THRESHOLD:
                delta_n2 += 1
                if pix1[i,j] < EDGE_DETECT_THRESHOLD:
                    X_in += 1
            # second image remove the edge
            if pix1[i,j] >= EDGE_DETECT_THRESHOLD:
                delta_n1 += 1
                if pix2[i,j] < EDGE_DETECT_THRESHOLD:
                    X_out += 1

    ECR = max(X_in/delta_n2, X_out/delta_n1)
    VALUE_FOR_PLOT.append(ECR)
    if DRY:
        return False
    return check_fade_or_wipe(ECR, index)
            
    
def answer(expand = True):
    global VIDEO
    with open(f"{VIDEO}_ground.txt", 'r') as f:
        shot_change = []
        for l in f.readlines()[4:]:
            if l.find('~') >= 0:
                if expand:
                    r = l.split('~')
                    for rr in range(int(r[0]), int(r[1].replace('\n', ''))+1):
                        shot_change.append(rr-1)
                else:
                    shot_change.append(l.strip('\n'))
            else:
                shot_change.append(int(l.replace('\n', '')))

    return shot_change

def shot_change_detect(compare=pair_wise):
    for i in range(FRAME_AMOUNT):
        if i % 100 == 0:
            print(f"{i:04}/{FRAME_AMOUNT}")
        if compare(f"{FILE_PREFIX}{i:04}.{FILE_EXTENSION}", f"{FILE_PREFIX}{i+1:04}.{FILE_EXTENSION}", i):
            print(i+1)
            ANSWER.append(f"{i+1}")
            EXPAND_ANSWER.append(f"{i+1}")

def draw_plot():
    global VIDEO
    global VALUE_FOR_PLOT

    y = VALUE_FOR_PLOT
    ans = answer()

    an_y = [sum(y)/len(y) for a in ans]
    new_y = [y[x] for x in ans]
    plt.plot(y)
    plt.plot(ans, an_y, 'ro')
    plt.show()

def draw_PR_plot():
    global ANSWER
    global EXPAND_ANSWER
    TRUE_ANSWER = answer(False)
    TRUE_EXPAND_ANSWER = answer()

    TP = len(set(ANSWER) & set(TRUE_ANSWER))
    FP = len(set(ANSWER) - set(TRUE_ANSWER))
    FN = len(set(TRUE_ANSWER) - set(ANSWER))

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument('-p', '--pair_wise', action="store_true")
    method_group.add_argument('-gh', '--grey_histogram', action="store_true")
    method_group.add_argument('-ch', '--color_histogram', action="store_true")
    method_group.add_argument('-l', '--likelihood_ratio', action="store_true")
    method_group.add_argument('-e', '--edge_detection', action="store_true")
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument('-v1', '--news', action="store_true")
    video_group.add_argument('-v2', '--soccer', action="store_true")
    video_group.add_argument('-v3', '--ngc', action="store_true")

    parser.add_argument('-d', '--dry', action = 'store_true', help="only print image")

    args = parser.parse_args()
    cmp_func = ''
    if args.pair_wise:
        cmp_func = pair_wise
    elif args.grey_histogram:
        cmp_func = histogram_comparison
    elif args.color_histogram:
        cmp_func = color_histogram_comp
    elif args.likelihood_ratio:
        cmp_func = likelihood_ratio
    elif args.edge_detection:
        cmp_func = edge_detection

    if args.news:
        FRAME_AMOUNT = 1379
        FILE_PREFIX = "news_out/news-000"
        FILE_EXTENSION = "jpg"
        VIDEO = "news"
        TWIN_COMPARISON_Tb = 42
        TWIN_COMPARISON_Ts = 42
    elif args.soccer:
        FRAME_AMOUNT = 864
        FILE_PREFIX = "soccer_out/soccer000"
        FILE_EXTENSION = "jpg"
        VIDEO = "soccer"
    elif args.ngc:
        FRAME_AMOUNT = 1059
        FILE_PREFIX = "ngc_out/ngc-"
        FILE_EXTENSION = "jpeg"
        VIDEO = "ngc"

    if args.dry:
        DRY = True

    shot_change_detect(cmp_func)
    draw_plot()
