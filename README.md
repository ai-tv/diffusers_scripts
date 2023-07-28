# diffusers Latent Couple

从底层代码来看 Latent Couple 实现
一点也不神秘！！！

我们通常见到的 Latent Couple 界面是这样的：
[图片]

其对应的 Prompts 是这样的
[图片]
一堆数字，冒号，逗号，都是啥啊，什么鬼啊？
Divisions Posoitions 以及 Weights 都只是帮助使用 webUI 的人画出下面这 3 张可视化图的，但是写成这样属实是一坨屎！

## 实际上这个东西代码实现非常简单
只用知道两个东西
  1. 目标区域的划分方式，完全可以不用是矩形，任意形状都可以
  2. 划分后各子区域的权重，需保证 latent space 中各个像素点的权重加和为 1

## 现在来用伪代码实现上述 webUI 中 latent couple
1. 构建 3 张（不一定是 3 张，看你的 prompts 有几条，不过多了效果会变差）mask 图
  1. Prompt A 对应的 maskA：一个 shape=(1,4,64,64) 全为 weight A 的 tensor
  torch.from_numpy(np.ones((1,4,64,64), dtype=np.float16) * weight_A).to("cuda")
  2. Prompt B1 对应的 maskB1：一个 shape=(1,4,64,64)，左边为 weight B1，右边为 0 的 tensor
  torch.from_numpy(np.concatenate((np.ones((1,4,64,32), dtype=np.float16), np.zeros((1,4,64,32), dtype=np.float16)), axis=-1) * weight_B1).to("cuda")
  3. Prompt B2 对应的 maskB2：一个 shape=(1,4,64,64)，左边为 0，右边为 weight B2 的 tensor
  torch.from_numpy(np.concatenate((np.zeros((1,4,64,32), dtype=np.float16), np.ones((1,4,64,32), dtype=np.float16)), axis=-1) * weight_B2).to("cuda")
2. 采样循环中把这些加进去，在单个采样 step 中：
L = pred_text(PromptA) * maskA + pred_text(PromptB1) * maskB1 + pred_text(PromptB2) * maskB2


TODO future
对齐 diffusers 和 webui
拆解单 lora
