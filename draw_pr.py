import matplotlib.pyplot as plt
import numpy as np


def add_line(file_name, label):
    pr = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for l in lines:
            items = l.strip('\n').split(' ')
            pr.append(items)

    # pr = np.sort(np.array(pr, dtype=float))
    # pr = np.array(sorted(pr, key=lambda x: x[0]), dtype=float)
    pr = np.array(pr, dtype=float)
    plt.plot(pr[:, 1], pr[:, 0], 'o', label=label)


add_line("pr_curve_news_gh.txt", 'news')
add_line("pr_curve_soccer_gh.txt", 'soccer')
# add_line("pr_curve_ngc_p.txt", 'ngc')
plt.xlabel("recalls")
plt.ylabel("precise")
plt.title("Pair wise")
plt.legend()
plt.show()
