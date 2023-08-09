# 将pth网络导出为onnx格式
import torch
import matplotlib.pyplot as plt
from DDPG import Actor

input_size = 2
output_size = 1
model = Actor(input_size,output_size)
checkpoint = torch.load('./checkpoint v1.pth')
model.load_state_dict(checkpoint['model_state_dict_actor'])
model.eval()
dummy_input = torch.randn(2)
input_names = ['sigma','delta_q']
output_names = ['Action']
torch.onnx.export(model,dummy_input,'model.onnx',verbose=True,input_names=input_names,output_names=output_names)
model = torch.jit.trace(model, dummy_input)
torch.save({
            'model_state_dict': model.state_dict(),
            },'model.pth')