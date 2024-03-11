> This code of fg_predictor follows [CPR](https://github.com/flyinghu123/CPR).
# Environment Install
```shell
git clone https://github.com/amazon-science/patchcore-inspection.git
cd patchcore-inspection
pip install  -e .
```
# Data Prepare
- copy the mvtec datset into the directory named `mvtec`.
    ```shell
    # directory tree
    data
    └── mvtec
        ├── bottle
        ├── cable
        ├── capsule
        ├── carpet
        ├── grid
        ├── hazelnut
        ├── leather
        ├── metal_nut
        ├── pill
        ├── screw
        ├── tile
        ├── toothbrush
        ├── transistor
        ├── wood
        └── zipper
    ```
 
# run

> python foreground_predictor.py
- create a directory named `data`
- copy the mvtec datset source `mvtec` into the directory named `data`.
- run the script `foreground_predictor.py`
- change the foreground path on [copy.bash](./copy.bash) line 3.
- run code `sh copy.bash`
