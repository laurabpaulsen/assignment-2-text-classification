
source ./env/bin/activate

echo "[INFO]: Classifying real and fake news. This may take a while..."

model="mlp logistic"

for m in $model; do
    echo "[INFO]: Training $m classifier..."
    python src/binary_text_classification.py --clf_type $m --save_models True
done

# deactivate the virtual environment
deactivate