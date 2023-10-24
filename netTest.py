# 网络验证
import torch
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体，例如黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像时负号'-'显示为方块的问题

from DDPG import Actor
from attackEnvironment import AttackEnv
from scipy.spatial import distance
import json
import os
import re


def find_max_checkpoint(directory, prefix, extension):
    max_checkpoint_num = -1
    max_checkpoint_file = ""

    # 遍历目录，找出所有符合条件的文件
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(extension):
            # 使用正则表达式从文件名中提取数字
            checkpoint_num = int(re.search(r"(\d+)\.pth$", filename).group(1))

            if checkpoint_num > max_checkpoint_num:
                max_checkpoint_num = checkpoint_num
                max_checkpoint_file = filename

    return max_checkpoint_file, max_checkpoint_num


input_size = 2
output_size = 2
actorModel = Actor(input_size, output_size)

# 使用函数
directory = "v911/" + "checkpoints/"
prefix = "checkpoint"
extension = ".pth"
max_checkpoint_file, max_checkpoint_num = find_max_checkpoint(
    directory, prefix, extension
)
target_bool = True
if max_checkpoint_num != -1:
    checkpoint_path = os.path.join(directory, max_checkpoint_file)
    checkpoint = torch.load(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    print("No checkpoint found.")

if target_bool:
    actorModel.load_state_dict(checkpoint["model_state_dict_actor_target"])
else:
    actorModel.load_state_dict(checkpoint["model_state_dict_actor"])


actorModel.eval()

iters = 10
env = AttackEnv()
episode = []
R_err = []

for i in range(iters):
    # 场景初始化
    xM = 0 + np.random.uniform(-500, 500)
    yM = 3000 + np.random.uniform(-500, 500)
    xT = 8000 + np.random.uniform(-500, 500)
    yT = 0
    vM = 200 + np.random.uniform(-50, 50)
    theta = np.random.uniform(np.deg2rad(-25), np.deg2rad(-15))
    r = distance.euclidean((xM, yM), (xT, yT))
    q = np.arctan((yT - yM) / (xT - xM))
    sigma = theta - q
    R = r / vM
    sim_step = 0.1
    misDistance = 0.1
    state = env.resetTest(sigma, R, q, sim_step)

    R_ = []
    sigma_ = []
    t_ = []
    test_ = []
    terr_ = []
    a_ = []
    q_ = []

    done = False
    with torch.no_grad():
        while not done:
            Action = actorModel(torch.as_tensor(state, dtype=torch.float32))
            state, done, R, sigma, q, acc, t = env.StepTest(Action)
            a_.append(acc * vM / 10)
            R_.append(R)
            sigma_.append(sigma)
            t_.append(t)
            q_.append(q)

        plt.figure(1)
        plt.plot(t_, sigma_)
        plt.title("速度前置角")
        plt.xlabel("时间/秒")
        plt.ylabel("角度/度")
        plt.figure(2)
        plt.plot(t_, R_)
        plt.title("R")
        plt.figure(3)
        plt.plot(t_, a_)
        plt.title("控制过载")
        plt.xlabel("时间/秒")
        plt.ylabel("过载/g")
        plt.figure(4)
        plt.plot(t_, [i * vM for i in R_])
        plt.title("弹目相对距离")
        plt.xlabel("时间/秒")
        plt.ylabel("距离/米")
        plt.figure(5)
        plt.plot(t_, q_)
        plt.title("视线角")
        plt.xlabel("时间/秒")
        plt.ylabel("角度/度")
        plt.tight_layout()
        episode.append(i + 1)
        R_err.append(R * vM)

plt.figure(6)
plt.scatter(episode, [abs(i) for i in R_err], s=10, alpha=0.3, marker=".")
plt.title("终端脱靶误差")
plt.xlabel("试验轮次")
plt.ylabel("距离/米")

for i in range(5):
    plt.figure(i + 1)
    plt.grid()
    plt.savefig(f"fig{i}{i}.png")

plt.show()
