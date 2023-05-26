# Stereo Vision Pipeline

## Description
This code contains a Python implementation of a stereo vision pipeline that estimates depth information from a pair of rectified stereo images. The pipeline involves feature matching, fundamental matrix estimation, camera pose recovery, image rectification, and disparity map computation.

## Requirements
To run the code, you will need the following Python libraries:

- OpenCV
- NumPy
- Matplotlib
- tqdm
You can install these libraries using pip:

`pip install opencv-python opencv-contrib-python numpy matplotlib tqdm`

## Usage
1. The dataset contains stereo images and calibration parameters. The expected dataset structure is:
```
- dataset_name/
    - calib.txt
    - im0.png
    - im1.png
```
2. The calib.txt file should contain the calibration parameters for both cameras, the disparity range, and the image dimensions.
3. Run the stereo vision pipeline on your dataset:
        `python stereo_vision.py --dataset <dataset_name>`
    example:
        `python stereo_vision.py --dataset artroom`
        `python stereo_vision.py --dataset chess`
        `python stereo_vision.py --dataset ladder`
4. If the python code exits with an except error "**Error: Run the code again!! could not compute R and T" run the code again to obtain the rotation and translation matrices.
4. The pipeline will output the disparity and depth maps, both in grayscale and color. The output images will be saved in the results folder with the naming convention disparity_depth_{dataset_name}.png.
5. The matching window process(SSD) takes approximately 15 minutes for completion.