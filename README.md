This is a incomplete pytorch implementation, please see tf branch for original implementation and models.

## installation
For environment setting up:
```
pip install -r requirement.txt
cd ./models/arch/dcn
python setup.py develop
cd ./models/arch/carafe
python setup.py develop
```

# Dataset Preparation
For testing data: please arange the test frames as the following format:

-path_to_training_set
    -source
        -vid1
            -vid1_000.png
            -vid1_001.png
            -vid1_002.png
            ...
        -vid2
        ...

## Testing
In the yaml config file:

modify:
DATA.PATH_TO_TEST_SET: "path to your prepared test set"
TEST.CHECKPOINT_FILE_PATH: "path to downloaded checkpoint"
TEST.OUTPUT_DIR: "path to your output dir"


