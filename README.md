# Permutation Training
This repository serves as the source code of our paper 

> Cai, Yongqiang, Gaohang Chen, and Zhonghua Qiao. "Neural networks trained by weight permutation are universal approximators." _Neural Networks_ (2025): 107277.
> 
> https://www.sciencedirect.com/science/article/pii/S089360802500156X?via%3Dihub
> 
> https://doi.org/10.1016/j.neunet.2025.107277

If you have any questions or interest for our work, feel free to cantact us by sending email to gaohang.chen@connect.polyu.hk.

---

**Requirement:** This project mainly relies on the following libraries:

- torch>=2.1.1
- numpy>=1.26.4
- jupyter-server>=2.15
- tqdm>=4.65.0
- matplotlib>=3.9.2
- tensorboardx>=2.2

---

The code is mainly inspired by the algorithm _LaPerm_ proposed by

> Qiu, Yushi, and Reiji Suda. "Train-by-reconnect: Decoupling locations of weights from their values." _Advances in Neural Information Processing Systems_ 33 (2020): 20952-20964.
> https://proceedings.neurips.cc/paper/2020/hash/f0682320ccbbb1f1fb1e795de5e5639a-Abstract.html

**Beware:** Our code here is primarily intended to verify our theoretical results, so the relevant setup and implementation are relatively basic. However, we believe this is a very interesting algorithm with promising potential, and hope to see more efficient implementations in the future.
