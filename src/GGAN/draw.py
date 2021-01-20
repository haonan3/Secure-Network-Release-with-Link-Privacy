import matplotlib.pyplot as plt
import numpy as np



models = ['Original','GGAN (no DP)',
          'GVAE','NetGAN',#'GraphRNN',
          'DPGGAN\nepsilon=10', 'DPGGAN\nepsilon=1', 'DPGGAN\nepsilon=0.1']



mean_imdb = [0.8661,0.7743,
             0.7714,0.7619,
             0.5931,0.5889,0.5798]
mean_dblp = [0.6824,0.6637,
             0.7463,0.6536,
             0.5527,0.5328,0.5137]
y = range(7)

plt.figure(figsize=(6,3))
bar_color=plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, 2))

bar1 = plt.barh(y=[i + 0.2 for i in y], height = 0.4,width=mean_dblp,
               alpha = 0.8, color = bar_color[0],label = 'DBLP')

bar2 = plt.barh(y=[i - 0.2 for i in y],height =0.4,width = mean_imdb,
               alpha = 0.8,color =bar_color[1],label = 'IMDB')

plt.yticks(y,models)
plt.xlim(0.5,0.9)
plt.ylabel('Models')
plt.xlabel('Accuracy')
plt.legend()
plt.tight_layout()


plt.savefig("link_pred.png", format='png', dpi=200, bbox_inches='tight')
plt.show()