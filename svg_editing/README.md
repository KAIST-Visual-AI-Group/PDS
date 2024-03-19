# SVG Editing with PDS

## Installation
Our SVG editing code is based on [ximinng/VectorFusion-pytorch](https://github.com/ximinng/VectorFusion-pytorch/).


Install `diffvg` library:
```
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
python setup.py install
```

## Run
To edit a svg with PDS, run the following command:
```
python pds_svg.py --svg_path {svg_file} --src_prompt "{source_prompt}" --tgt_prompt "{target_prompt}"
```


