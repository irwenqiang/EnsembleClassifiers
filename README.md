EnsembleClassifiers
===================

LibD3C: Ensemble classifiers with a clustering and dynamic selection strategy

LibD3C是基于聚类的动态选择循环集成分类器

分类器选择阶段：
使用K-means/Affinity Propagation，从1000多个未经过参数调优的弱分类器中选择出少量的，差异度大(最小错分交集)的弱分类器；

分类器集成阶段：
在分类器集成阶段，才用包括爬山策略、集成前序选择、集成后续选择等集成方法集成差异度大且集成分类精度高的弱分类器集合


If you use LibD3C please cite the following paper:
LibD3C: Ensemble classifiers with a clustering and dynamic selection strategy Chen Lin, Wenqiang Chen, Cheng Qiu, Yunfeng Wu, Sridhar Krishnan, Quan Zou
