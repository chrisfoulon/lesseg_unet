checkpoint_root_path=$1

for noise_type in "gibbs" "rician" "bias" "all_noises";
do
  for incr in {0..10};
  do
    for i in {0..4};
    do
      lesseg_unet -o "/media/chrisfoulon/DATA1/a_imagepool_mr/noise_increment_experiment/noise_increment/${noise_type}_$((incr * 10))/fold_${i}" \
      -li "/media/chrisfoulon/HDD2/final_training_set/split_lists_per_fold/fold_${i}_img_list.csv" \
      -lli "/media/chrisfoulon/HDD2/final_training_set/split_lists_per_fold/fold_${i}_lbl_list.csv" \
      -trs "destructive_${noise_type}_$((incr * 10))" -nw 3 -mt swin-unetr -overlap -sa \
      -pt "${checkpoint_root_path}/fold_${i}"\
      -oss;
    done;
  done;
done;