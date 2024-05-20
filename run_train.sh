echo "Running Python script..."
python train.py --style_name city_center --content_weight 1 --style_weight 80000 --tv_weight 0.00001 --device cuda:1 &
PID=$!
echo "PID of Python script: $PID"
wait $PID  # Optionally wait for the Python script to complete
echo "Script completed."
