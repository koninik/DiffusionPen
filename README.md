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


<p align="center">
  <img src="imgs/diffusionpen.png" alt="Overview of the proposed DiffusionPen" style="width: 60%;">
</p>

<p align="center">
  Overview of the proposed DiffusionPen
</p>

## 🚀 Models on Hugging Face 🤗
You can download the pre-trained models from HF by clicking here: <a href="https://huggingface.co/konnik/DiffusionPen">https://huggingface.co/konnik/DiffusionPen</a> 

## Training from scratch

To train the diffusion model run:
```
python train.py 
```

## Sampling - Regenerating IAM

If you want to regenerate the full IAM training set you can run:
```
python 
```

## Sampling - Single image

If you want to generate a single word with a random style you can run:
```
python sampling.py 
```

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

