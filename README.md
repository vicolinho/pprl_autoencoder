## Autoencoder evaluation for Bloom filter encryption  

### single encoder model

**Setup**
<pre>
Data Owner A &larr;&mdash; parameters &mdash;&rarr; Data Owner B  
     &darr;                              &darr;  
Bloom Filters                 Bloom Filters  
     &darr;                              |  
train Encoder  &mdash;&mdash;&mdash; Encoder &mdash;&mdash;&mdash;&rarr;     | 
     &darr;                              &darr;  
encode Bloom Filters          encode Bloom Filters
     |                              |
      &mdash;&mdash;&mdash;&mdash;&rarr;    Linkage Unit    &larr;&mdash;&mdash;&mdash;&mdash;
</pre>
running *single_model.py&nbsp; -cdir&nbsp; \<config_directory\>*&nbsp; in &nbsp;*/src/*&nbsp; will do the following for each configuration file in &nbsp;*/src/\<config_directory\>/*&nbsp;:
 - build an autoencoders of the structure specified in the configuration file and fit them on *Data Owner A*'s set of Bloom-Filters
 - encode the two datasets using the fitted encoder, normalize the encoded data
 - for each record in *B*'s encoded Data, search for the nearest neighbour in *A*s encoded data. If the distance is below a certain threshold (specified in the configuration file), the two datapoints are considered a match.  

the linkage results, as well as the encoder model and the training progress data, for a configuration file &nbsp;*/src/\<config_directory\>/\<configname\>.json*&nbsp; are stored in &nbsp;*/src/\<config_directory\>/\<configname\>/* .

### separate encoder model
**Setup**
<pre>
Data Owner A &larr;&mdash; parameters &mdash;&rarr; Data Owner B  
     &darr;                              &darr;  
Bloom Filters                 Bloom Filters  
     &darr;                              &darr;  
train Encoder                 train Encoder  
     &darr;                              &darr;  
encode Bloom Filters          encode Bloom Filters
     |                              |
      &mdash;&mdash;&mdash;&mdash;&rarr;    Linkage Unit    &larr;&mdash;&mdash;&mdash;&mdash;
</pre>
**Linkage Mapper Data Generation**
<pre>
                Data Owner A  &larr;&mdash; b_decode(D) &mdash;&mdash;  Data Owner B  
                      |                              &uarr;
a_encode(b_decode(D)) |                              | random Data D
                       &mdash;&mdash;&mdash;&mdash;&rarr;    Linkage Unit    &mdash;&mdash;&mdash;&mdash;&mdash;
                                     &darr;
                      pairs (d, a_encode(b_decode(d)))
</pre>
running *run_all_configs.py&nbsp; -cdir&nbsp; \<config_directory\>*&nbsp; in &nbsp;*/src/*&nbsp; will do the following for each configuration file in &nbsp;*/src/\<config_directory\>/*&nbsp;:
 - build two autoencoders of the same structure (specified in the configuration file) for two data owners *A*,*B*&nbsp; and fit them on their respective sets of Bloom-Filters
 - encode the two datasets using the fitted encoders, normalize the encoded data
 - generate training data in order to build a mapper between the two encodings. This is done as follows:
  - a random dataset is sampled from an n-dimensional standard normal distribution in the linkage unit (n being the dimension of the encodings) and sent to data owner *B*
  - the datapoints are transformed to fit the the output distribution of *B*s encoder and fed into the decoder network
  - the decoder outputs are sent to *A*, fed into *A*s encoder network, the encoder outputs are normalized and sent to the linkage unit
 - the linkage unit trains a mapper model on the obtained pairs of datapoints
 - the datasets are linked by applying the mapper to *B*s encoded datapoints and searching for the nearest neighbor of the output in *A*s encoded data. If the distance is below a certain threshold (specified in the configuration file), the two datapoints are considered a match.  

the linkage results, as well as all models, generated datasets and training progress data, for a configuration file &nbsp;*/src/\<config_directory\>/\<configname\>.json*&nbsp; are stored in &nbsp;*/src/\<config_directory\>/\<configname\>/* .  

Results of the evaluation run as part of the bachelor thesis are stored in &nbsp;*/src/configs/configurations_\<dataset\>\[_single\]/* .
