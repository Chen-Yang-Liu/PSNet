<div align="center">

<h1><a href="https://ieeexplore.ieee.org/document/10283451">Progressive Scale-aware Network for Remote sensing Image Change Captioning</a></h1>

**[Chenyang Liu](https://chen-yang-liu.github.io/), [Jiajun Yang](https://levir.buaa.edu.cn/members/index.html), [Zipeng Qi](https://levir.buaa.edu.cn/members/index.html), [Zhengxia Zou](https://scholar.google.com.hk/citations?hl=en&user=DzwoyZsAAAAJ), and [Zhenwei Shi*✉](https://scholar.google.com.hk/citations?hl=en&user=kNhFWQIAAAAJ)**


</div>

## Welcome to our repository! 

This repository contains the PyTorch implementation of the paper: "Progressive Scale-aware Network for Remote sensing Image Change Captioning". 

For more information, please see our published paper in [[IEEE]([https://ieeexplore.ieee.org/document/10271701](https://ieeexplore.ieee.org/document/10283451))]  ***(Accepted by IGARSS 2023)***

### Data preparation
Firstly, download the image pairs of LEVIR_CC dataset from the [[Repository](https://github.com/Chen-Yang-Liu/RSICC)]. 
Then preprocess dataset as follows:
```python
python create_input_files.py --karpathy_json_path path/Levir-CC-dataset/LevirCCcaptions.json --image_folder path/Levir-CC-dataset/images 
```
After that, you can find some resulted files in `./data/`. 
Of course, you can use our provided resulted  files directly in [[Hugging face](https://huggingface.co/lcybuaa/PSNet/tree/main)].


### Train
Make sure you performed the data preparation above. Then, start training as follows:
```python
python ./train.py --encoder_image vit_b_32 --data_folder ./data/ --savepath ./checkpoints/5-times/
```

### Evaluate
You can download our pretrained model in [[Hugging face](https://huggingface.co/lcybuaa/PSNet/tree/main)]. Put the model in `./checkpoints/5-times/`, then run
```python
python ./eval.py --encoder_image vit_b_32 --data_folder ./data/ --model_path ./checkpoints/5-times/
```
We recommend training 5 times to get an average score.
## Citation: 
```
@INPROCEEDINGS{10283451,
  author={Liu, Chenyang and Yang, Jiajun and Qi, Zipeng and Zou, Zhengxia and Shi, Zhenwei},
  booktitle={IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={Progressive Scale-Aware Network for Remote Sensing Image Change Captioning}, 
  year={2023},
  volume={},
  number={},
  pages={6668-6671},
  doi={10.1109/IGARSS52108.2023.10283451}}
```




