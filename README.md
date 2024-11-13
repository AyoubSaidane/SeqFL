# SeqFL Setup Instructions

This README provides the steps to set up the SeqFL environment with all necessary dependencies for your project.

## Prerequisites

Ensure that you have [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) installed on your system.

## Steps to Set Up

### 1. Create a Conda Environment

Create a new Conda environment named `SeqFL` with Python 3.8:

```bash
conda create --name SeqFL python=3.8 -y
```

### 2. Activate the Environment

Activate the newly created environment:

```bash
conda activate SeqFL
```

### 3. Install Core Dependencies

Install essential libraries such as `numpy`, `scipy`, `tqdm`, `matplotlib`, and `scikit-learn`:

```bash
conda install numpy scipy tqdm matplotlib scikit-learn -y
```

### 4. Install PyTorch and CUDA Support

Install PyTorch, torchvision, torchaudio, and CUDA support for GPU acceleration (compatible with CUDA 12.1):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 5. Install TensorBoard

Install TensorBoard to monitor model training:

```bash
conda install -c conda-forge tensorboard -y
```


