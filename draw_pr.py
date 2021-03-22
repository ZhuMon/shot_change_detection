import matplotlib.pyplot as plt
import numpy as np
import sys


def add_line(file_name, label):
    pr = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        # items = lines[0].strip('\n').split(' ')
        # pr = np.array([items], dtype=float)
        for l in lines:
            items = l.strip('\n').split(' ')
            # remove same value in precise
            # if float(items[1]) not in pr[:,1]:
                # new = np.array(items, dtype = float)
                # pr = np.vstack((pr, new))
            pr.append(items)

    # pr = np.sort(np.array(pr, dtype=float))
    pr = np.array(sorted(pr, key=lambda x: x[1]), dtype=float)
    # pr = np.array(pr, dtype=float)
    plt.plot(pr[:, 1], pr[:, 0], '-', label=label)


# add_line("pr_curve/news_gh.txt", 'news')
# add_line("pr_curve/soccer_gh.txt", 'soccer')
# add_line("pr_curve/ngc_gh.txt", 'ngc')
add_line("pr_curve/ngc_gh.txt", 'grey level histogram')
# add_line("pr_curve/ngc_ch.txt", 'color level histogram')
add_line("pr_curve/ngc_p.txt", 'pair wise')
add_line("pr_curve/ngc_l.txt", 'likelihood')
add_line("pr_curve/ngc_e.txt", 'edge detection')
plt.xlabel("recalls")
plt.ylabel("precise")
plt.title("ngc")
plt.legend()
plt.show()
