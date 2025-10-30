# How to run

## Download model
```
modelscope download --model XiaomiMiMo/MiMo-7B-Base
mv ~/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-7B-Base /data/llm/
```

## Replace modeling
```
cp MiMo-7B/modeling_mimo.py /data/llm/MiMo-7B-Base/
```

## Run inference
```
python multi_token_prediction.py
```
