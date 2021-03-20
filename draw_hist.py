import matplotlib.pyplot as plt

y = []
with open("soccer_hist_color", 'r') as f:
    for l in f.readlines():
        # y.append(float(l.split(' ')[1]))
        y.append(float(l.replace('\n', '')))

answer = []
with open("soccer_ground.txt", 'r') as f:
    for l in f.readlines()[4:]:
        if l.find('~') >= 0:
            r = l.split('~')
            for rr in range(int(r[0]), int(r[1].replace('\n', ''))+1):
                answer.append(rr-1)

an_y = [10 for a in answer]
new_y = [y[x] for x in answer]
# plt.plot(answer, new_y, 'o')
plt.plot(y)
plt.plot(answer, an_y, 'ro')
plt.show()

# print(*new_y)
# print(*answer)
