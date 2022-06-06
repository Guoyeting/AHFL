### Privacy vs. Efficiency: Achieving Both through Adaptive Hierarchical Federated Learning

This repository includes source code for the paper Y. Guo, F. Liu, T. Zhou, Z. Cai and N. Xiao. "Privacy vs. Efficiency: Achieving Both through Adaptive Hierarchical Federated Learning".

#### Requirements

The code runs on Python 3. To install the dependencies, run
```
pip3 install -r requirements.txt
```
#### Dataset supported
Then, download the datasets manually and put them into the `datasets` folder.
- For MNIST dataset, download from <http://yann.lecun.com/exdb/mnist/> and put the standalone files into `datasets/mnist`.
- For CIFAR-10 dataset, download the "CIFAR-10 binary version (suitable for C programs)" from <https://www.cs.toronto.edu/~kriz/cifar.html>, extract the standalone `*.bin` files and put them into `datasets/cifar-10-batches-bin`.

#### Training
- To simulate the cloud datacenter in the paper, run this command `python server.py` and wait until you see `Waiting for incoming connections...` in the console output.
- To simulate the network edge (including the edge server and the end devices), run this command `python client.py`.
- You will see console outputs on both the server and clients indicating message exchanges. The code will run for a few minutes before finishing.

#### Code Structure

- `config.py` stores all configuration options. In this file, you can configure the constraints of our optimization problem, i.e. resource budgets and privacy budgets of the edge and the cloud, etc. 
And you can also configure the setup, including the dataset, the number of end devices, etc. 

- Directory `control_algorithm` includes the adaptive control algorithms in our paper.  

- Directory `data_reader` simulates different data distribution cases, including iid and non-iid settings. 

- Directory `datasets` stores the datasets we used. Other datasets can also applied through some slight modifications. 

- Directory `models` stores the network structure of training models. Other models can also applied through some slight modifications.

- Directory `statistic` is used to record our experimental results in the Directory `results`.

- Directory `util` contains some auxiliary functions, like sending and receiving messages, calculating the privacy costs.


