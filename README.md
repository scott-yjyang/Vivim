# Vivim
[IEEE TCSVT] Vivim: a Video Vision Mamba for Medical Video Segmentation 

[[arXiv](https://arxiv.org/abs/2401.14168)] [[Dataset](https://drive.google.com/file/d/1mRkvqcaic-0W8WnDRLUWRuzrcp7f8dWc/view?usp=sharing)]



## News
- 25-04-21. Vivim has been accepted by TCSVT. Welcome to try our dataset!
- 24-08-01. Upload several example cases of VTUS dataset
- 24-03-11. ❗❗Update on Code. Welcome to taste.😄
- 24-02-08. Update on Method and Experiments.
- 24-01-26. This project is still quickly updating 🌝. Check TODO list to see what will be released next.
- 24-01-25. The paper has been released on arXiv.



## A Quick Overview 

<img width="600" height="400" src="https://github.com/scott-yjyang/Vivim/blob/main/assets/framework1.png">


## Environment Setup
### Clone this repository and navigate to the root directory of the project.

```bash
git clone https://github.com/scott-yjyang/Vivim.git

cd Vivim
```

### Install basic package

```bash
conda env create -f environment.yml
```


### Install casual-conv1d

```bash
cd causal-conv1d

python setup.py install
```

### Install mamba

```bash
cd mamba

python setup.py install
```


### TODO LIST

- [x] Release Model
- [x] Release training scripts
- [x] Release evaluation
- [x] Release Ultrasound dataset
- [ ] Experiments on other video object segmentation datasets.
- [ ] configuration



## Thanks

Code is based on [hustvl/Vim](https://github.com/hustvl/Vim), [bowang-lab/U-Mamba](https://github.com/bowang-lab/U-Mamba).

## Cite
If you find it useful, please cite and star
~~~
@article{yang2024vivim,
  title={Vivim: a Video Vision Mamba for Medical Video Object Segmentation},
  author={Yang, Yijun and Xing, Zhaohu and Zhu, Lei},
  journal={arXiv preprint arXiv:2401.14168},
  year={2024}
}
~~~
