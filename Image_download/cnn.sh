#!/bin/sh
array=("SGD" "RMSprop" "Adagrad" "Adadelta" "Adam" "Adamax" "Nadam")
for opti in ${array[@]}
do
    for ep in 10 20 30 40 50
    do
        python3 cnn.py $opti $ep
    done
done
