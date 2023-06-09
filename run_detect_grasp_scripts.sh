#!/bin/sh

if [ "$1" = "detect" ]
then
    cd Detic
    python demo_detic.py --image-dir=$2 --vocabulary=$3 --custom-vocabulary=$4 --device=$5
elif [ "$1" = "grasp" ]
then
    cd ../grasping
    python eval.py --scene-dir=$2 --threshold=$3 --scene-type=$4 --obj-id=$5
fi
