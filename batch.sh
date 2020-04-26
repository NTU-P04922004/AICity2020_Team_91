WORKSPACE_PATH=/home/kuohsin/workspace/dataset/AIC20_track4/train-data-videos
START_IDX=1
END_IDX=100
PRETRAINED_MODEL_PATH=/home/kuohsin/workspace/detectron2_lab/output/exp_lr_2e-2_cascade_mask_rcnn_input_size/model_0069999.pth

for ((i=START_IDX; i<=END_IDX; i++))
do
    echo "Processing video $i..."
    . ./video_convert.sh $WORKSPACE_PATH/$i.mp4 1
    echo "$i: video extraction done"
    python3 background_analysis.py $WORKSPACE_PATH/$i $WORKSPACE_PATH
    echo "$i: background analysis done"
    python3 roi_analysis.py $WORKSPACE_PATH/test_fg_imgs/$i $WORKSPACE_PATH/video_masks/$i
    echo "$i: roi analysis done"
    python3 detect_cars.py $WORKSPACE_PATH/test_bg_imgs/$i $WORKSPACE_PATH/video_masks $WORKSPACE_PATH $PRETRAINED_MODEL_PATH
    echo "$i: car detection done"
    python3 calculate_anomaly_events.py $WORKSPACE_PATH/bg_detections/bg_test_$i.txt $WORKSPACE_PATH/test_bg_imgs/$i >> result.txt
    echo "$i: anomaly events calculation done"
    echo "Processing video $i done."

    # rm -rf $WORKSPACE_PATH/$i
    # # rm -rf $WORKSPACE_PATH"/test_bg_imgs/"$i
    # rm -rf $WORKSPACE_PATH/test_fg_imgs/$i
done

echo "All done!"
