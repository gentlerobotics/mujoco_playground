# Installation

Playground libraries are currently not shipped with cuda12 wheels for 5080. Must install those directly:

```bash
conda create -n mj_playground python=3.10 -y
conda activate mj_playground
pip install mujoco
pip install mujoco_mjx
pip install brax
pip install playground
pip uninstall -y jax jaxlib jax_cuda12_plugin nvidia-cublas-cu12     # clean out the CPU build
pip install --upgrade pip setuptools wheel          # fresh tooling

# Pulls in jax, jaxlib (CPU+XLA), and the CUDA 12 plugin in one shot
pip install "jax[cuda12]==0.6.0"   
pip install --force-reinstall "jax==0.6.0" "jaxlib==0.6.0" "jax-cuda12-plugin==0.6.0"
```