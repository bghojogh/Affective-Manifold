# Affective Manifolds

Using affective manifolds, we can model a machine's mind to feel like a human. 

This is the code for the following paper:
- Benyamin Ghojogh, "Affective Manifolds: Modeling Machine's Mind to Like, Dislike, Enjoy, Suffer, Worry, Fear, and Feel Like A Human", arXiv:2208.13386, 2022.
- Link to the paper: https://arxiv.org/abs/2208.13386

# How to run the code

For running the code of affective manifolds:
- Install the required packages:
```bash
pip install -r requirements.txt
```
- Put the dataset files `train.csv` and `test.csv` in the path `./dataset/`.
- Change the config file in the path `./configs/config.json`. You can use the example config files in the path `./configs/example_configs/`.
- Run the code `main_affective.py`:
```bash
python main_affective.py
```
 
The part of code for Siamese network is inspired by the code `main_siamese.py`, which is taken from the following Kaggle page:
- https://www.kaggle.com/code/hirotaka0122/triplet-loss-with-pytorch/notebook
