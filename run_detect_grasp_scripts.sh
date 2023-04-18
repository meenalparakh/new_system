#!/bin/sh

if [ "$1" = "detect" ]
then
    cd ../Detic
    python demo.py --scene-dir=$2 --num-views=$3 --data-lst-fname=$4
elif [ "$1" = "grasp" ]
then
    cd ../grasping
    python eval.py --scene-dir=$2 --threshold=$3 --scene-type=$4 --obj-id=$5
fi