 # 🔥 DiffusionPen: Towards Controlling the Style of Handwritten Text Generation

 <p align='center'>
  <b>
    <a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11492_ECCV_2024_paper.php">ECCV Paper</a>
    |
    <a href="http://www.arxiv.org/abs/2409.06065">ArXiv</a>
    |
    <a href="https://drive.google.com/file/d/1BXHPPpjD84mhdYUnnHeXCc-A-3tWhkaR/view?usp=share_link">Poster</a>
    |
    <a href="https://huggingface.co/konnik/DiffusionPen">Hugging Face</a>
      
  </b>
</p> 



## 📢 Introduction
- We introduce DiffusionPen, a few-shot diffusion model developed for generating stylized handwritten text. By using just a few reference samples (as few as five), it learns a writer’s unique handwriting style and generates new text that imitates that style.
- DiffusionPen effectively captures both seen and unseen handwriting styles with fewer examples. This is achieved through a style extraction module that combines metric learning and classification, allowing for greater flexibility in representing various writing styles.
- We evaluated DiffusionPen on IAM and GNHK (only qualitative) handwriting datasets, demonstrating its ability to generate diverse and realistic handwritten text. The generated data closely matches the real handwriting distribution, leading to enhancement in Handwriting Text Recognition (HTR) systems when used for training. 

<p align="center">
  <img src="imgs/diffusionpen.png" alt="Overview of the proposed DiffusionPen" style="width: 60%;">
</p>

<p align="center">
  Overview of the proposed DiffusionPen
</p>

## 🚀 Download Dataset & Models from Hugging Face 🤗
You can download the pre-processed dataset and model weights from HF here: <a href="https://huggingface.co/konnik/DiffusionPen">https://huggingface.co/konnik/DiffusionPen</a> 

- IAM pre-processed dataset in .pt for direct loading in <a href="https://huggingface.co/konnik/DiffusionPen/tree/main/saved_iam_data">saved_iam_data</a>
- Style weights for the style encoder (also DiffusionPen-class and DiffusionPen-triplet) in <a href="https://huggingface.co/konnik/DiffusionPen/tree/main/style_models">style_models</a>
- DiffusionPen weights for IAM in <a href="https://huggingface.co/konnik/DiffusionPen/tree/main/diffusionpen_iam_model_path/models">diffusionpen_iam_model_path/models</a>

Place the folders 📁`saved_iam_data`, 📁`style_models`, and 📁`diffusionpen_iam_model_path` in the main code directory.

For VAE encoder-decoder and DDIM we use <a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5">stable-diffusion-v1-5</a>.


## 🧪 Sampling using DiffusionPen

For single image sampling run
```
python train.py --save_path ./diffusionpen_iam_model_path --style_path ./style_models/iam_style_diffusionpen.pth --train_mode sampling --sampling_mode single_sampling
```

For paragraph sampling run
```
python train.py --save_path ./diffusionpen_iam_model_path --style_path ./style_models/iam_style_diffusionpen.pth --train_mode sampling --sampling_mode paragraph
```

We also provide the IAM training and validation set images generated using **DiffusionPen** in the following link:  
[Download IAM Dataset Generated with DiffusionPen](https://drive.google.com/file/d/1IcQLZ8yIqdLgYyZUsFOl3v8qYN3h2RJL/view?usp=share_link)
(test set will be soon uploaded!!!)

## 🏋️‍♂️ Train with Your Own Data

If you'd like to train DiffusionPen using your own data, simply adjust the data loader to fit your dataset and follow these 2 steps:

1. Train the Style Encoder:
```
python style_encoder_train.py
```
2. Train DiffusionPen:
```
python train.py --epochs 1000 --model_name diffusionpen --save_path /new/path/to/save/models --style_path /new/path/to/style/model.pth --stable_dif_path ./stable-diffusion-v1-5
```

## 📝 Evaluation

We compare **DiffusionPen** with several state-of-the-art generative models, including [GANwriting](https://github.com/omni-us/research-GANwriting), [SmartPatch](https://github.com/MattAlexMiracle/SmartPatch), [VATr](https://github.com/aimagelab/VATr), and [WordStylist](https://github.com/koninik/WordStylist). 
The Handwriting Text Recognition (HTR) system used for evaluation is based on [Best practices for HTR](https://github.com/georgeretsi/HTR-best-practices).


---

## 📄 Citation

If you find our work useful for your research, please cite:

```bibtex
@article{nikolaidou2024diffusionpen,
  title={DiffusionPen: Towards Controlling the Style of Handwritten Text Generation},
  author={Nikolaidou, Konstantina and Retsinas, George and Sfikas, Giorgos and Liwicki, Marcus},
  journal={arXiv preprint arXiv:2409.06065},
  year={2024}
}

