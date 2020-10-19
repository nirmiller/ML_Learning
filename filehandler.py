
#only works with latest python

import splitfolders

input = "Dataset\Garbage"
output = "Dataset\processed_data"
splitfolders.ratio(input, output, seed=42, ratio=(.6, .2, .2))


