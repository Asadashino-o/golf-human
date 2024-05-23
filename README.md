# golf-human-detection

![](https://github.com/Asadashino-o/golf-human/blob/main/sample_videos/example.gif)

## Overview

This project is an attempt to perceive changes in the position of the human buttocks and head during golf through AI. It mainly relies on the densepose project in Detectron2, a project package of Facebook AI Research, and also uses Flask to write a visual webpage connecting the front-end and server back-end.It is forked from Flode-Labs/vid2densepose.

## Prerequisites

To utilize this tool, ensure the installation of:
- Python 3.8 or later
- PyTorch (preferably with CUDA for GPU support)

## Installation Steps

1. Create a virtual environment for your anaconda or miniconda:
   ```
   conda create -n your_env_name python=3.9
   conda activate your_env_name
   
   ```
   Install the right pytorch version and able to use cuda.
   You can use "nvcc -V" to see your cuda version

2. Clone the repository:
    ```bash
    git clone https://github.com/Asadashino-o/golf-human.git
    cd golf-human
    ```
   
3. Install necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Clone the Detectron repository:
    ```bash
    git clone https://github.com/facebookresearch/detectron2.git
    ```
   
5. Download the pkl weight file to human-golf/
   Download URL:
   https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl

## Usage Guide

Run the script:
    
```bash
python main.py -i sample_videos/input_video.mp4 -o sample_videos/output_video.mp4
```

The script processes the input video and generates an output video with the draw-line format.

####  Web Interface
You can also use the flask to run the web with an interface. To do so, run the following command:
```bash
cd flaskk
python app.py
```
After it runs in linux command-line.
You can open your browser and input your URL.

## Acknowledgments

Thanks to:
- Facebook AI Research (FAIR) for the development of DensePose.
- The contributors of the Detectron2 project.
- [Gonzalo Vidal](https://www.tiktok.com/@_gonzavidal) for the sample videos.
- [Sylvain Filoni](https://twitter.com/fffiloni) for the deployment of the Gradio Space in [Hugging Face](https://huggingface.co/spaces/fffiloni/video2densepose).
- https://github.com/Flode-Labs

It is just a copy-learning project.

## Support

For any inquiries or support, please file an issue in GitHub repository's issue tracker.

