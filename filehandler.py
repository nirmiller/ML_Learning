
#only works with latest python

import splitfolders

input = "FlowerDataSet\Flowers"
output = "FlowerDataSet\processed_data"
splitfolders.ratio(input, output, seed=42, ratio=(.6, .2, .2))


