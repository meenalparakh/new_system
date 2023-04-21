pip install 'git+https://github.com/facebookresearch/detectron2.git'

git clone https://github.com/facebookresearch/Detic.git --recurse-submodules

cd ./Detic
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

git submodule init
git submodule update
pip install -r requirements.txt

cp ../demo_detic.py demo_detic.py
