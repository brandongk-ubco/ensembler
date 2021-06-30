rm config.yaml
setx DATA_DIR "E:\work\datasets"
setx NUM_WORKERS 15
echo %DATA_DIR%
python trainer.py --data.dataset cityscapes --model.out_classes 19
