import matplotlib.pyplot as plt
import numpy as np

class plotAttack:
    def __init__(self):
        self.fig, self.axs = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 1, 1,1]})  # 创建一个3x1的子图数组

    def plot_subgraphs(self,data1, data2, data3,data4,location):
        self.axs[0].plot(data1[0],data1[1])
        self.axs[1].plot(data2[0],data2[1])
        self.axs[2].plot(data3[0],data3[1])
        self.axs[3].plot(data4[0],data4[1])
        self.fig.tight_layout()
        self.fig.savefig(location)
    
    def plot_reset(self):
        for ax in self.axs:
            ax.cla()
        self.axs[0].set_title('Trajectory')
        self.axs[1].set_title('Acceleration')
        self.axs[2].set_title('sigma')
        self.axs[3].set_title('q')

if __name__ =="__main__":
    # 创建一些示例数据
    data1 = np.random.rand(2,100)
    data2 = np.random.rand(2,100)
    data3 = np.random.rand(2,100)
    ppa = plotAttack()
    ppa.plot_subgraphs(data1,data2,data3)


