#  The Effect of Differential Privacy on Recommendation

This is the official repository of the paper *The Effect of Differential Privacy on Recommendation* submitted to 
CIKM 2022.
This work investigates the effect on recommender systems' performance of the application of **differential privacy**.
Our analysis focuses on three well-known datasets from three different domains: 
[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) (films), 
[LibraryThing](https://cseweb.ucsd.edu/~jmcauley/datasets.html#social_data) (books) and 
[AmazonDigitalMusic](https://jmcauley.ucsd.edu/data/amazon/) (music).

## Download the Datasets
All the datasets are available within the repository, in the 'data/' directory.

## Reproduce the Experiments

For reproducing our results several steps are needed. 
- [Binarize](#binarize) the datasets
- From the binarized version [generate](#generate-datasets-and-apply-randomized-response) 600 sub-datasets and apply **randomized response** on them, using different 
privacy budgets.
- [Train](#training) four different recommender models on each of the generated 600 sub-datasets
- [Collect](#collect-results) the results and [compute](#generate-results-differences) the variations of the selected metrics (Precision and ARP)
- Train the [regressor model](#compute-regression-and-store-results) and store the **weights** and variables **p-values**

Note, the pipeline above could be applied on the three datasets independently. 
In the following the commands for each dataset are reported.

### Binarize

Binarize the dataset, transforming explicit feedbacks in implicit.
#### MovieLens1M
```
python binarize_dataset.py --dataset data/MovieLens1M/MovieLens1M.tsv --result data/MovieLens1M --threshold 3
```
#### LibraryThing
```
python binarize_dataset.py --dataset data/LibraryThing/LibraryThing.tsv --result data/LibraryThing --threshold 6
```
#### Amazon Digital Music
```
python binarize_dataset.py --dataset data/AmazonDigitalMusic/AmazonDigitalMusic.tsv --result data/AmazonDigitalMusic --threshold 3 --header 0
```

### Generate Datasets and Apply Randomized Response

From the binarized datasets, generate 600 sub-datasets. Then, apply randomized response on each of them.

#### MovieLens1M
```
python generate_and_randomize_data.py --dataset data/MovieLens1M/binarized/MovieLens1M.tsv --result data/MovieLens1M --n 600
```
#### LibraryThing
```
python generate_and_randomize_data.py --dataset data/LibraryThing/binarized/LibraryThing.tsv --result data/LibraryThing --n 600
```
#### Amazon Digital Music
```
python generate_and_randomize_data.py --dataset data/AmazonDigitalMusic/binarized/AmazonDigitalMusic.tsv --result data/AmazonDigitalMusic --n 600
```


### Training

Train the recommender models on each one of the generated sub-datasets and their randomized versions.

#### MovieLens1M
```
python run_experiments.py --dataset MovieLens1M --start 0 --end 600 --eps 3 2 1 0.5 --baseline
```
#### LibraryThing
```
python run_experiments.py --dataset LibraryThing --start 0 --end 600 --eps 3 2 1 0.5 --baseline
```
#### Amazon Digital Music
```
python run_experiments.py --dataset AmazonDigitalMusic --start 0 --end 600 --eps 3 2 1 0.5 --baseline
```

### Collect Results

Collect the recommendation results stored by Elliot and store them at 'performance/'.

#### MovieLens1M
```
python collect_results.py --dataset MovieLens1M --output performance
```
#### LibraryThing
```
python collect_results.py --dataset LibraryThing --output performance
```
#### Amazon Digital Music
```
python collect_results.py --dataset AmazonDigitalMusic --output performance
```


### Generate Results Differences

From the previously collected results, compute the differences between the performance obtained on the original 
dataset and the noised versions. The results are augmented with the selected datasets' characteristics 
and stored at 'performance/delta/'.
```
python generate_results.py a--dataset MovieLens1M LibraryThing AmazonDigitalMusic --subfolder performance --model Random ItemKNN MostPop Random RP3beta EASER --metrics Precision ARP --characteristics space_size shape gini_item ratings_per_user ratings_per_item --output performance/delta --eps 0.5 2 3
```


### Compute Regression and Store Results

Compute the regressors. It is needed a command for storing the weights of the model and a command for the pvalues.
The results are stored at 'performance/regression/'

#### MovieLens1M
```
python regression.py --dataset MovieLens1M --dir performance/delta --metric Precision ARP --result performance --target params
python regression.py --dataset MovieLens1M --dir performance/delta --metric Precision ARP --result performance --target pvalues
```
#### LibraryThing
```
python regression.py --dataset LibraryThing --dir performance/delta --metric Precision ARP --result performance --target params
python regression.py --dataset LibraryThing --dir performance/delta --metric Precision ARP --result performance --target pvalues
```
#### Amazon Digital Music
```
python regression.py --dataset AmazonDigitalMusic --dir performance/delta --metric Precision ARP --result performance --target params
python regression.py --dataset AmazonDigitalMusic --dir performance/delta --metric Precision ARP --result performance --target pvalues
```

## License
This work is released under [APACHE2 License](./LICENSE).

## Acknowledgements
Our datasets are constructed thanks to 
- [Grouplens](https://grouplens.org/) research lab for MovieLens dataset
- [LibraryThing](https://www.librarything.com/) website for the LibraryThing dataset
- [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/) for the Amazon Digital Music dataset
