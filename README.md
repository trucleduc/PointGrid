# PointGrid: A Deep Network for 3D Shape Understanding

## Prerequisites:
	Python (with necessary common libraries such as numpy, scipy, etc.)
	TensorFlow
	You need to prepare your data in *.mat file with the following format:
		'points': N x 3 array (x, y, z coordinates of the point cloud)
		'labels': N x 1 array (1-based integer per-point labels)
		'category': scalar (0-based integer model category)

## Train:
	python train.py
## Test:
	python test.py

If you find this code useful, please cite our work at <br />
<pre>
@article{PointGrid, <br />
	author = {Truc Le and Ye Duan}, <br />
	titile = {{PointGrid}: A Deep Network for 3D Shape Understanding}, <br />
	journal = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, <br />
	month = {June}, <br />
	year = {2018}, <br />
} <br />
</pre>
