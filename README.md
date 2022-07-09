<h1 align=center> <b>c4der</b> </h1>

<p align="center">
<a href="https://www.python.org/downloads/release/python-3100/" 
target="_blank"><img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python Version" /></a>
<img src="https://img.shields.io/github/license/g0bel1n/TinyAutoML?style=flat-square" alt="Licence MIT" />
</p>

---

<p align="center">
c4der is an experimental library. <br/>
It stands for Clustering for Dynamic Event Recognition<br/>
contact : lucas.saban[at]ensae.fr<br/>
WIP
</p>

---

## Description

Object tracking and domain expertise allows people to build event recognition software. Yet, in many situations, object tracking cannot be achieved because object visual obstruction, low sampling frequency, low object recognition reliability. 

c4der offers a solution tailored for event recognition in 2D videos. 
Its core is Density Based for Spatial Clustering Applications with Noise (DBSCAN)combined with time dimensionnality, adaptable cluster density distance (see [1]), non-spatial features. 

It goes further by unveiling stochastic cluster propagation using a Gaussian kernel and minimal intra-cluster variance threshold. 

On a more pratical level, it allows time-split clustering.


## References

    [1] : Birant, Derya, and Alp Kut. 
    "ST-DBSCAN: An algorithm for clustering spatial–temporal data." 
    Data & Knowledge Engineering 60.1 (2007): 208-221.
