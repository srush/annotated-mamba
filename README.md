Part 1: 

Blog: https://srush.github.io/annotated-mamba/hard.html

Notebook: https://github.com/srush/annotated-mamba/blob/main/Scan.ipynb


Mamba: Linear-Time Sequence Modeling with Selective State Spaces
https://arxiv.org/abs/2312.00752


## Challenge

The triton version is still a lot slower than the mamba custom kernel. If anyone has ideas for speeding it up, I would love to hear them. The file `final.py` has a minimal version with benchmarking. 

```python
pip install mamba-ssm
pip install -U http://kermit.bounceme.net:8900/triton-3.0.0-cp310-cp310-linux_x86_64.whl
```
