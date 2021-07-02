import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.CMS)

BP = [10, 11, 12, 13, 14, 15, 16, 18, 20]
AUC = [0.47932853191410546, 0.5364737352283813, 0.7331836928738131, 0.8406868725534988, 0.89508242470642, 0.9154525949890718, 0.9255248015015497, 0.9315163985033896, 0.9308775607226207]

plt.figure(figsize=(6,5))
plt.plot(BP, np.array(AUC), '-o', lw=2)
plt.hlines(0.9549295839867626, 10, 20, colors='gray', linestyles='dashed')
plt.ylim(0.4, 1.0)
plt.xlabel('Total bit precision')
plt.ylabel('AUC')
#plt.title('AUC vs Total bit precision')
plt.legend()
plt.legend(['hls4ml', 'Expected (0.955)'], title='112 nodes, 148 edges', loc='lower right')
plt.tight_layout()
plt.savefig('AUC_vs_BP.pdf')
