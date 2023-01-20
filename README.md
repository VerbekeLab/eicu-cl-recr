# Client Recruitment for Federated Learning in eICU LoS prediction

## Important
The FedML dependency for this work requires the FedML package to be installed from the forked repository. The source code is changed to allow for federated training and metric reporting specifically for the regression setting as required for eICU Length of Stay predictions (LoS). Installation can be done using pip as follows:

    ```
    python -m pip install 'git+https://github.com/vscheltjens/FedML.git@master#subdirectory=python'
    ```

## Installing the dataset
In this work the publicly available eICU dataset was used as it contains hospital identifiers allowing to map data to the originating hospital, an interesting feature to simulate a (near) real-world environment for federated learning of a neural network.

Instructions for downloading and installing the dataset can be found here:
    ```
    Link to eICU db to be included
    ```

Note that we recommend installing the database in a `container environment`, i.e. with both `postgresql` and `pgadmin4` running in containers orchestrated by docker.

## Extracting and preprocessing data
The data extraction process is similar to the extraction and preprocessing pipeline as proposed by rocheteau. In addition to rocheteau's pipeline, the code in this repo is altered such that also the corresponding hospitalid's are extracted and 0-indexed. This is necessary as we treat data originating from different hospitals as indiviual data silo's

Once the database has been set up, place the contents of the extraction directory on the docker container and cd in `data_extraction`. From here you can:

1. Connect to the database:
    ```
    psql 'dbname=eicu user=eicu options=--search_path=eicu_crd'
    ```

2. Extract the data and store under a folder `./eICU_Data`
    ```
    \i create_all_tables.sql
    ```

3. Run the preprocessing 
    ```
    python -m data_extraction.run_all_preprocessing 
    ```


## Installing fedml from the forked repo
To reproduce the experiments you should install fedml from the forked repository which contains alterations to the source code needed for training a Federated GRU model that predicts Length of Stay based on the first 24hours of patient data into ICU. 

Installing fedml from the forked repository:

1. create a virtual environment using conda
    ```
    conda create -n env_name python=3.10
    ```

2. Install the dependencies
    ```
    pip install requirements.txt
    ```

3. Install fedml from the forked repository 
    ```
    python -m pip install 'git+https://github.com/vscheltjens/FedML.git@master#subdirectory=python'
    ```


## Reproducing experiments
To reproduce the experiments as reported in the paper run the following commands from within the `root directory`

### Central training
This treats the data as one global dataset irrespective of the originating hospital. 

### Federated training
1. Naive model with all clients
    - Note that this requires a significant amount of compute
    - Update the following parameters in `config.yaml`
        ```
        client_num_in_total: 189
        client_num_per_round: 189
        comm_round: 10
        epochs: 2
        recruitment: False
        ```
    - Run the command:
        ```
        python -m distributed_training.federated --cf 'absolute_path_to_config.yml'
        ```

2. Standard implementation of FedAvg with client selection (randomly sampled) for each round of communication
    - Update the following parameters in `config.yaml`
        ```
        client_num_in_total: 189
        client_num_per_round: 10
        comm_round: 20
        epochs: 2
        recruitment: False
        ```

    - Run the command:
        ```
        python -m distributed_training.federated --cf 'absolute_path_to_config.yml'
        ```
3. Model with all recruited clients
    - Update the following parameters in `config.yaml`
        ```
        client_num_in_total: 189
        client_num_per_round: 189
        comm_round: 10
        epochs: 2
        recruitment: True
        ```

    - Run the command:
        ```
        python -m distributed_training.federated --cf 'absolute_path_to_config.yml'
        ```

4. Model with client selection amongst all recruited clients
    - Update the following parameters in `config.yaml`
        ```
        client_num_in_total: 189
        client_num_per_round: 10
        comm_round: 20
        epochs: 2
        recruitment: True
        ```
        
    - Run the command:
        ```
        python -m distributed_training.federated --cf 'absolute_path_to_config.yml'
        ```

Note that if you wish to track training with wandb, you should update the tracking_args in `config.yml` accordingly 
