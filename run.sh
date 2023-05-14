
source ./env/bin/activate

echo "[INFO]: Classifying real and fake news. This may take a while..."

model="mlp logistic"
vec="tfidf count"

for m in $model; do
    for v in $vec; do
        echo "[INFO]: Training $m classifier with $v vectorizer"
        python src/binary_text_classification.py --clf_type $m --save_models True --vec_type $v
    done
done

# deactivate the virtual environment
deactivate