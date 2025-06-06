'''
    @User: LG
    @Auther: http://www.xlonggang.cn
    @Date: 2025/6/6 0006 17:28 
    @Description: 
    @Project ：NLP_Project 
    @File    ：tools.py
    @version: 1.0
'''
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
