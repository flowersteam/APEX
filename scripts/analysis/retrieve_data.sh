#!/usr/bin/env bash

tar_torso() {
    experiment=$1
    id=$2
    path="/home/pi/apex_results/$experiment"
    tar_path="/home/pi/apex_results/$experiment-$id.tar"
    local_tar_path="/tmp/$experiment-$id.tar"

    echo -e "Archiving data from experiment $experiment on apex-$id..."
    ssh torso-$id "cd $path && find . -maxdepth 4 -name '*.pickle' | sudo tar -cf $tar_path --files-from -"
    if [ $? -eq 0 ]; then
        echo -e "Tared on torso $id"
    else
        echo -e "\e[31mapex-$id FAILED while creating tar archive on remote!\e[39m"
    fi
}
export -f tar_torso

tar_torso_all() {
    echo -e "Starting to tar all"
    experiment=$1
    for id in `seq 1 6`; do
        tar_torso "$experiment" "$id" &
    done
    wait
    echo -e "Finished to tar all"
}
export -f tar_torso_all


retrieve_tar() {
    experiment=$1
    id=$2
    tar_path="/home/pi/apex_results/$experiment-$id.tar"
    local_path="/data/APEX/$1"
    mkdir -p $local_path
    scp torso-$id:$tar_path $local_path/torso-$id.tar
    if [ $? -eq 0 ]; then
        echo -e "Retrieved tar $id"
    else
        echo -e "Failed to retrieved tar $id"
    fi
}
export -f retrieve_tar

retrieve_tar_all() {
    echo -e "Starting to retrieve all tars"
    experiment=$1
    for id in `seq 1 6`; do
        retrieve_tar "$experiment" "$id" &
    done
    wait
    echo -e "Finished Tar all"
}
export -f retrieve_tar_all


untar_torso() {
    experiment=$1
    local_path="/data/APEX/$1"
    cd $local_path
    tar xf $local_path/torso-$id.tar
    if [ $? -eq 0 ]; then
        echo -e "Untared $id"
    else
        echo -e "Failed to Untar $id"
    fi
}
export -f untar_torso

untar_torso_all() {
    echo -e "Starting Untar all"
    experiment=$1
    for id in `seq 1 6`; do
        untar_torso "$experiment" "$id" &
    done
    wait
    echo -e "Finished Untar all"
}
export -f untar_torso_all


clean_working_tree() {
    echo -e "Starting to clean working tree"
    experiment=$1
    local_path="/data/APEX/$1"
    data_path="$local_path/media/usb0/$1"
    for task in $data_path/*/; do
        #echo "task $task"
        for condition in $task*/; do
            condition=$(basename $condition)
            #echo "condition $task$condition"
            for trial in $task$condition/*/; do
                #echo "trial $trial"
                # Check iterations
                cd $trial
                n_iterations="$(ls *.pickle | wc -l)"
                cd $local_path
                echo "$trial #iterations: $n_iterations"
                if [ "$n_iterations" -ge "$2" ]; then
                    # Find new path
                    mkdir -p $local_path/$condition
                    n_trials="$(ls -l $local_path/$condition | grep ^d | wc -l)"
                    echo "number of trials of $condition : $n_trials"
                    new_path=$local_path/$condition/trial_$n_trials
                    mkdir $new_path
                    # Move files to new path
                    cd $trial
                    cp *.pickle $new_path
                    cd $local_path
                fi
            done
        done
    done
    echo -e "Finished to clean working tree"
}
export -f clean_working_tree


merge_experiments() {
    echo -e "Starting merging $1 $2"
    experiment_from="$1"
    experiment_to="$2"
    local_path="/data/APEX/$1"
    data_path="$local_path/media/usb0/$1"
    target_path="/data/APEX/$2"
    for task in $data_path/*/; do
        echo "task $task"
        for condition in $task*/; do
            condition=$(basename $condition)
            #echo "condition $task$condition"
            for trial in $task$condition/*/; do
                #echo "trial $trial"
                # Check iterations
                cd $trial
                n_iterations="$(ls *.pickle | wc -l)"
                cd $local_path
                echo "$trial #iterations: $n_iterations"
                if [ "$n_iterations" -ge "$3" ]; then
                    # Find new path
                    mkdir -p $target_path/$condition
                    n_trials="$(ls -l $target_path/$condition | grep ^d | wc -l)"
                    echo "number of trials of $condition : $n_trials"
                    new_path=$target_path/$condition/trial_$n_trials
                    mkdir $new_path
                    # Move files to new path
                    cd $trial
                    cp *.pickle $new_path
                    cd $local_path
                fi
            done
        done
    done
    echo -e "Finished merging $1 in $2"
}
export -f merge_experiments



tar_torso_all "$1"
retrieve_tar_all "$1"
untar_torso_all "$1"

# Only one of the 2 following:
#merge_experiments "$1" "$2" 20000
#clean_working_tree "$1" 20000

#rm -r /data/APEX/$1/media

#cd ~/scm/Flowers/APEX/scripts/analysis
#python analyze_files.py $1


