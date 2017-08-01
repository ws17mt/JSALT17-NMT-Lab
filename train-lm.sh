echo "Training the model..."

WORK=$HOME/nlp/jsalt/JSALT17-NMT-Lab
mkdir -p models

# Training
#rm -rf models/multi30k-$1-$2/baseline #remove the existing one if required!
if [ -z ${SOCKEYE+x} ]; then
  echo "No tokenizer is initialized. Source ENV.sh first";
fi

PYTHONPATH=$SOCKEYE python3 $SOCKEYE/sockeye/trainlm.py \
  --source $WORK/data/multi30k/train.en.atok \
  --target $WORK/data/multi30k/train.en.atok \
  --validation-source $WORK/data/multi30k/val.en.atok \
  --validation-target $WORK/data/multi30k/val.en.atok \
  --source-vocab vocab.en.json \
  --target-vocab vocab.en.json \
  --checkpoint-frequency 200 \
  --word-min-count 2 \
  --rnn-num-layers 1 \
  --rnn-cell-type lstm \
  --rnn-num-hidden 64 \
  --num-embed-source 64 \
  --num-embed-target 64 \
  --batch-size 20 \
  --normalize-loss \
  --dropout 0.65 \
  --optimizer adam \
  --use-cpu \
  --initial-learning-rate 0.001 \
  --output ./models/multi30k-lm-en
