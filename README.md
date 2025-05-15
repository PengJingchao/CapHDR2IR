# CapHDR2IR
Source code of the paper: **"[CapHDR2IR: Caption-Driven Transfer from Visible Light to Infrared Domain](https://arxiv.org/abs/2411.16327)"**.

## Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

Download the [caption branch weight](https://github.com/PengJingchao/CapHDR2IR/releases/tag/CaptionBranch) and put it into \<path to your project\>/models/modules/HDR2IR.

## Download the Dataset
Download the dataset from [HDRT dataset](https://huggingface.co/datasets/jingchao-peng/HDRTDataset) for training HDR-to-IR. 

If you want to test SDR-to-IR performance, please download [SDR validation data](https://github.com/PengJingchao/CapHDR2IR/releases/tag/SDR2IR_Val_Data).


## Training

To start training, run:

```bash
python train.py
```

## Testing

To test performance, run:

```bash
python test_RGB2IR.py
```

## Citation

If you find this project useful, please cite our work:

    @article{HDRT2025,
      title={HDRT: A Large-Scale Dataset for Infrared-Guided HDR Imaging},  
      author={Jingchao Peng and Thomas Bashford-Rogers and Francesco Banterle and Haitao Zhao and Kurt Debattista},
      year={2025},
      journal={Information Fusion},
      volume = {120},
      pages = {103109},
      year = {2025},
      issn = {1566-2535},
      doi = {10.1016/j.inffus.2025.103109},
    }

    @misc{peng2024caphdr2ircaptiondriventransfervisible,
      title={CapHDR2IR: Caption-Driven Transfer from Visible Light to Infrared Domain}, 
      author={Jingchao Peng and Thomas Bashford-Rogers and Zhuang Shao and Haitao Zhao and Aru Ranjan Singh and Abhishek Goswami and Kurt Debattista},
      year={2024},
      eprint={2411.16327},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.16327}, 
    }
