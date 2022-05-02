import matplotlib.pyplot as plt
import sys
import math
from hist_sent_len import *

for key in histogram_data:
    if len(histogram_data[key]) == 0:
        continue
    plt.title(f"{key}")
    if "train" in key:
        y = [math.log(x,10) for x in list(histogram_data[key].values())]
        plt.ylabel("log of Count base 10")
    else:
        y = list(histogram_data[key].values())
        plt.ylabel("Count")
    x = list(histogram_data[key].keys())
    plt.xlabel("Sentence length")
    plt.bar(x, y)
    if sys.argv[4] != "0":
        plt.savefig(f"{sys.argv[1]}/tmp/{key}.png")
    else:
        plt.savefig(f"{sys.argv[1]}/{key}.png")
    plt.clf()

