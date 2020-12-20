# Photo Popularity Predictor

> Take a folder of photos and assign a score to each of them

If you hate trying to find the best photo to post on insta out of 200 look alike photos you took today, this is for you.

I used a trained ML model written by [Intrinsic-Image-Popularity](https://github.com/dingkeyan93/Intrinsic-Image-Popularity), a PyTorch implementation of the research paper [Intrinsic Image Popularity Assessment](https://arxiv.org/abs/1907.01985).

### How to Run

1. Clone this repo and `cd` there
2. Put your photos under `/pic`. If you have a different folder, make sure to put the directory when running `python test.py --folder_path=[PATH]`
3. `python test.py --help`
