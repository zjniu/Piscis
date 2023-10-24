# Notebooks
This folder contains the Jupyter notebooks used to obtain our results. Here, you will find several sub-folders, each
prefixed with a unique numerical value. This prefix indicates the sequence in which the folders, and subsequently the
notebooks within them, should be accessed and executed. All code blocks in each Jupyter notebook should be executed in a
sequential order.

The outputs of these notebooks are saved in the `outputs` folder. If you would like to reproduce our results from
scratch, you may delete the `outputs` folder before running the notebooks. You may start with either step 0 or 1,
depending on whether you want to directly generate the Piscis dataset from the raw images and annotations or use the
pre-generated version from [Hugging Face](https://huggingface.co/datasets/wniu/Piscis). If you would like to only regenerate the panels used to create the
figures, you can skip steps 0-7 and directly execute the notebook `8_generate_panels/generate_panels.ipynb`. This will
override the pre-generated PDFs in the `outputs/panels` folder.