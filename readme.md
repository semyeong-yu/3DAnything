# prerequisites

## libraries
- lightning==2.5.1.post0
- open-clip-torch==2.0.2
- torch
    ```bash
    # please install torch version 2.3.1 with appropriate CUDA version
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ```
- please install other libraries if required 

## data
- [Processed Objaverse](https://huggingface.co/datasets/threeoe/3DDST)

## weights

### training purpose
- [Processed Weights](https://huggingface.co/JH-C-k/diffusion_project_weight)

### validation purpose
- [Trained Weights](https://huggingface.co/JH-C-k/diffusion_project_weight_trained)

# Training
- Please change the weight `path` in `normal_variant_i2t_multimodal_v3_256.yaml` appropriately after downloading the weights from `Processed Weights`. then run the following command in the `zero123` directory to train the model:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py -t --base /workspace/code/3DAnything/zero123/configs/normal_variant_i2t_multimodal_v3_256.yaml --gpus 4 --scale_lr False --load_composed_weights
```

# Validation
- please change the yaml path in `config_path` and weight path in `weight_path` of `result_renderer.ipynb` appropriately after downloading the weights from `Trained Weights`. then run the notebook to generate the results.