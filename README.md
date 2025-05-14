## SAM-adapter: Adapting SAM in Underperformed Scenes

## Environment
pip install -r requirements.txt


## Inference and Evaluation: The evaluation code is adapted from the CVPR paper "Learning Semantic Associations for Mirror Detection".
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
python evaluate.py -gt -pred
```

## Checkpoints
https://drive.google.com/drive/u/0/folders/15cOLOxxQnhK6OyhArhne6ZxNXz-SHmhV

## Dataset

### PMD and MSD Dataset: The PMD and MSD datasets can be downloaded from the EBLNet GitHub repository on the dataset page.
- **[PMD](https://github.com/hehao13/EBLNet)**
- **[MSD](https://github.com/hehao13/EBLNet)**

