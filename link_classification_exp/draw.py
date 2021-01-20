import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
from scipy import stats

models = ['Original', 'GGAN', 'DPGGAN']
styles = ['k1-', 'g1-', 'r.--'] #, 'g*--', 'b^--']
mean = {}
mean['Original'] = [[0.1, 1.0, 10.0], [0.8661, 0.8661, 0.8661]]
mean['GGAN'] = [[0.1, 1.0, 10.0], [0.6316, 0.6316, 0.6316]]
mean['DPGGAN'] = [[0.1, 1.0, 10.0], [0.5798, 0.5889, 0.5931]]

for m in models:
	plt.plot(mean[m][0], mean[m][1], styles[models.index(m)], label=m)

plt.grid(linestyle='--', linewidth=0.5)
plt.xlim(0, 10.1) # ind.
plt.ylim(0.5, 0.9) #.MSG

plt.xlabel('epsilon', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend(fontsize=10, loc='lower right', ncol=1)
plt.tight_layout()

plt.savefig("test.png", format='png', dpi=200, bbox_inches='tight')
plt.show()