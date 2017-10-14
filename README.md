# thalnet
Open-source implementation of ThalNet (https://arxiv.org/pdf/1706.05744.pdf)

---

## Oct 13th Hang Yu customized version
Added lazy_property, mnist data set. ThalNet still has tensorflow inner data structure to fix, but RNN and MLP works just fine.

## Oct 13th Hang Yu Mnist success!
All models are under 60,000 total parameters.
### MLP
![](/tensorboard/MLP.png)
### GRU 
![](/tensorboard/GRU.png)
### ThalNet-GRU 
![](/tensorboard/ThalNet_GRU.png)
### ThalNet-FF-GRU
![](/tensorboard/ThalNet_FF_GRU.png)
