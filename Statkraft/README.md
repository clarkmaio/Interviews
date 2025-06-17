# Statkraft Assessment 

This is the assesment for a Data Scientist position in Statkraft.

After this assessment I've never received a feedback (netiher positive or negative).

The original document with assesment specifications provided by Statkraft is `Statkraft/Technical Challenge - Quant Developers.pdf`.

The file `Statkraft.pdf/Statkraft_Assessment.pdf` is the document I've produced to present results. 


## Data
The data is stored in the folder `data/`. 
The file `processed_data.hdf` contains data already processed and ready to use.

In `data_manager.py` you will find the classes I've written to scrape most of the data.
BPA data is not scraped but downloaded from the website.

## How to run the code
The only script is supposed to run is `run_main.py`.

In `settings/parser.py` you will find the parser the main function will use to read the input arguments.

You should run the code from terminal. You can run it in two different modes:
1. Mode **analysis**: `python run_main.py --mode analysis`
2. Mode **forecast**: `python run_main.py --efmode forecast`

### Mode analysis
It just load data and create plots (the same used is the report).
Plot are saved in `.output/plots/` folder.

You can change the output directory from `settings/settings.yaml` file

### Mode forecast
It load data, fit models (all of them), compute performance and save results.
Few plots are also created.

Results are saved in `.output/results/` report as csv.



