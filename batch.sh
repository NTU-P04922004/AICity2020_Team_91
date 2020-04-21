DIR_PATH=/home/kuohsin/workspace/dataset/AIC20_track4/test-data
START_IDX=1
END_IDX=2
MODEL_WEIGHT_PATH=/home/kuohsin/workspace/detectron2_lab/output/exp_lr_2e-2_cascade_mask_rcnn_input_size/model_0069999.pth

for ((i=START_IDX; i<=END_IDX; i++))
do
    echo "Processing video $i..."
    . ./video_convert.sh $DIR_PATH/$i.mp4 5
    echo "$i: video extraction done"
    python3 background_analysis.py $DIR_PATH/$i $DIR_PATH
    echo "$i: background analysis done"
    python3 roi_analysis.py $DIR_PATH/test_fg_imgs/$i $DIR_PATH/video_masks/$i
    echo "$i: roi analysis done"
    python3 detect_cars.py $DIR_PATH/test_bg_imgs/$i $DIR_PATH $MODEL_WEIGHT_PATH
    echo "$i: car detection done"
    python3 calculate_anomaly_events.py $DIR_PATH/bg_test_$i.txt $DIR_PATH/test_bg_imgs/$i >> result.txt
    echo "$i: anomaly events calculation done"
    echo "Processing video $i done."

    # rm -rf $DIR_PATH/$i
    # # rm -rf $DIR_PATH"/test_bg_imgs/"$i
    # rm -rf $DIR_PATH/test_fg_imgs/$i
done

echo "All done!"
