# Neural Categorical Prior

The official repository accompanying the paper: 
"Neural Categorical Priors for Physics-Based Character Control" 
(https://tencent-roboticsx.github.io/NCP/).


## Installation
### Step 1: Install Isaac Gym
Begin by downloading Isaac Gym from its official [website](https://developer.nvidia.com/isaac-gym).
After downloading, carefully follow the provided installation
instructions to ensure Isaac Gym is properly set up on your system.

### Step 2: Install External Dependencies
With Isaac Gym successfully installed, the next step is to install 
the external dependencies required for this repository. 
Open your terminal or command prompt and navigate to the root directory of this repository. 
Then, execute the following command to install the necessary dependencies:
```
pip install -e .
```
This command installs the current directory (denoted by .) as a package in editable mode (-e), 
which means you can make changes to the source code and have them immediately affect 
your installed package without needing to reinstall.


Due to compatibility issues between Isaac Gym and newer versions of PyTorch, and the fact that older versions of PyTorch do not support torchrun, there are specific requirements for the PyTorch version in this project. We use PyTorch 1.12. Please install the corresponding version of PyTorch according to your CUDA version.

## Motion Visualization
To access the motion clips, navigate to the directory `ncp/data/motions/ncp_motions`. 
These clips are stored in the same format as [ASE](https://github.com/nv-tlabs/ASE), with individual motion clips saved as .npy files. The datasets for these motions are defined in .yaml files, which include a list of the motion clips to be incorporated into the dataset. To visualize any of these motion clips, you can use the command below:
#### Sword & Shield
``` 
python scripts/view_motion.py --num_envs 2 --cfg_env ncp/data/cfg/env/humanoid_sword_shield_tracking.yaml --motion_file ncp/data/motions/ncp_motions/dataset_sword_shield_tracking.yaml
``` 
#### Boxing
``` 
python scripts/view_motion.py --num_envs 2 --cfg_env ncp/data/cfg/env/humanoid_boxing_tracking.yaml --motion_file ncp/data/motions/ncp_motions/dataset_boxing_tracking.yaml
``` 
To configure the number of actors in the environment, the `--num_envs` argument is used. 
The `--cfg_env` argument allows for the specification of environment configurations for the task.
Additionally, the `--motion_file` argument is available for selecting a dataset of motion clips to visualize.
&nbsp;
## Imitation Policy

### Training
To train an imitation policy that imitates a dataset of motion clips, the following command can be executed:
#### Sword & Shield
```
python ncp/run.py --task HumanoidTracking --cfg_env ncp/data/cfg/env/humanoid_sword_shield_tracking.yaml --cfg_train ncp/data/cfg/train/ncp_sword_shield_tracking.yaml --motion_file ncp/data/motions/ncp_motions/dataset_sword_shield_tracking.yaml --num_envs 4096 --headless
```
#### Boxing
```
python ncp/run.py --task HumanoidTracking --cfg_env ncp/data/cfg/env/humanoid_boxing_tracking.yaml --cfg_train ncp/data/cfg/train/ncp_boxing_tracking.yaml --motion_file ncp/data/motions/ncp_motions/dataset_boxing_tracking.yaml --num_envs 4096 --headless
```
The `--cfg_train` argument specifies the training configurations for the policy, 
while the `--headless` option is used to disable visualizations during the training process.
### Testing

After the training process, the policy can be tested by executing the command below:
#### Sword & Shield
```
python ncp/run.py --task HumanoidTracking --cfg_env ncp/data/cfg/env/humanoid_sword_shield_tracking.yaml --cfg_train ncp/data/cfg/train/ncp_sword_shield_tracking.yaml --motion_file ncp/data/motions/ncp_motions/dataset_sword_shield_tracking.yaml --num_envs 2 --checkpoint ncp/data/model/ncp_models/ss_tracking.pth --test
```

#### Boxing
```
python ncp/run.py --task HumanoidTracking --cfg_env ncp/data/cfg/env/humanoid_boxing_tracking.yaml --cfg_train ncp/data/cfg/train/ncp_boxing_tracking.yaml --motion_file ncp/data/motions/ncp_motions/dataset_boxing_tracking.yaml --num_envs 2 --checkpoint ncp/data/model/ncp_models/boxing_tracking.pth --test
```
The `--checkpoint` argument should be assigned the path to the trained model. Pre-trained models are available in the `ncp/data/model/ncp_models/` directory.
&nbsp;

## Prior
### Training
After training the imitation policy, a prior can be learned to guide the decoder of the imitation policy by executing the command below:

#### Sword & Shield
```
python ncp/run.py --task HumanoidTrackingPrior --cfg_env ncp/data/cfg/env/humanoid_sword_shield_prior.yaml --cfg_train ncp/data/cfg/train/ncp_sword_shield_prior.yaml --motion_file ncp/data/motions/ncp_motions/ss_tracking_result.npy --num_envs 4096 --headless --pretrained_model_checkpoint ncp/data/model/ncp_models/ss_tracking.pth 
```
#### Boxing
```
python ncp/run.py  --task HumanoidTrackingPrior --cfg_env ncp/data/cfg/env/humanoid_boxing_prior.yaml --cfg_train ncp/data/cfg/train/ncp_boxing_prior.yaml --motion_file ncp/data/motions/ncp_motions/boxing_tracking_result.npy --num_envs 4096 --headless --pretrained_model_checkpoint ncp/data/model/ncp_models/boxing_tracking.pth
```
The `--pretrained_model_checkpoint` argument is used to specify the path to the imitation policy's pretrained model. 
The `--motion_file` argument specifies the motion dataset that the prior aims to perform. 
It's important to note that if the performance of the imitation policy does not closely match the motion dataset, 
a resultant motion file has to be recorded using the imitation policy itself. This allows for the refinement of 
the prior based on the actual outputs of the imitation policy, ensuring a closer alignment 
with the intended motion behaviors. Here, the files `ss_tracking_result.npy` and `boxing_tracking_result.npy` represent 
the tracking results for the `dataset_sword_shield_prior.yaml` and `dataset_boxing_prior.yaml`, respectively.

### Testing
After the training process is complete, you can test the prior by executing the following command:
#### Sword & Shield
```
python ncp/run.py --task HumanoidTrackingPrior --cfg_env ncp/data/cfg/env/humanoid_sword_shield_prior.yaml --cfg_train ncp/data/cfg/train/ncp_sword_shield_prior.yaml --num_envs 2 --checkpoint ncp/data/model/ncp_models/ss_prior.pth --test
```
#### Boxing
```
python ncp/run.py --task HumanoidTrackingPrior --cfg_env ncp/data/cfg/env/humanoid_boxing_prior.yaml --cfg_train ncp/data/cfg/train/ncp_boxing_prior.yaml --num_envs 2 --checkpoint ncp/data/model/ncp_models/boxing_prior.pth --test
```
## Upper-Level Policy
### Training
Once the imitation policy and prior have been trained, the upper-level policy can be trained to address new downstream 
tasks using the following code:
#### Sword & Shield
```
python ncp/run.py  --task HumanoidStrike --cfg_env ncp/data/cfg/env/humanoid_sword_shield_strike.yaml --cfg_train ncp/data/cfg/train/ncp_sword_shield_strike.yaml --num_envs 4096 --headless --pretrained_model_checkpoint ncp/data/model/ncp_models/ss_prior.pth
```
Note that the original work utilized [TLeague](https://github.com/tencent-ailab/TLeague) for training the multi-agent system. 
This repository serves as a reimplementation of the original code. 
Due to the complexity of reproducing prioritized fictitious self-play, training code is not provided here. 
However, the corresponding test code is available below.
### Testing
The upper-level policy can be tested using the following command:
#### Sword & Shield
```
python ncp/run.py --task HumanoidStrike --cfg_env ncp/data/cfg/env/humanoid_sword_shield_strike.yaml --cfg_train ncp/data/cfg/train/ncp_sword_shield_strike.yaml --num_envs 2 --test --checkpoint ncp/data/model/ncp_models/ss_strike.pth
```
#### Boxing
```
python ncp/run.py --task HumanoidBoxing --cfg_env ncp/data/cfg/env/humanoid_boxing.yaml --cfg_train ncp/data/cfg/train/ncp_boxing.yaml --num_envs 2 --checkpoint ncp/data/model/ncp_models/boxing_106.pth --opponent_checkpoint ncp/data/model/ncp_models/boxing_100.pth --test
```
For boxing, given its multi-agent nature, in addition to the `--checkpoint` argument that specifies the 
checkpoint for the self agent, an additional checkpoint for the opponent is required. 
This is done using the `--opponent_checkpoint` argument, which allows you to designate a
specific checkpoint for the opponent agent. Here, two models are provided: 
`boxing_100.pth` and `boxing_106.pth`. These correspond to the 100th and 106th models, 
respectively, from the self-play training process.

## Multi GPU Training
Similar to [rl_games](https://github.com/Denys88/rl_games), we utilize [torchrun](https://pytorch.org/docs/stable/elastic/run.html)
for enabling training across multiple GPUs. Additionally, the `--multi_gpu` argument is also required. For instance, 
to execute the training of an imitation policy using the 'Sword & Shield' dataset with two workers on a single node, 
the following command can be used:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 ncp/run.py --multi_gpu --task HumanoidTracking --cfg_env ncp/data/cfg/env/humanoid_sword_shield_tracking.yaml --cfg_train ncp/data/cfg/train/ncp_sword_shield_tracking.yaml --motion_file ncp/data/motions/ncp_motions/dataset_sword_shield.yaml --num_envs 4096 --headless
```

## Citation

If you find this repo useful for your research, please consider citing the paper:
```
@article{
        zhu2023NCP,
        author = {Zhu, Qingxu and Zhang, He and Lan, Mengting and Han, Lei},
        title = {Neural Categorical Priors for Physics-Based Character Control},
        year = {2023},
        issue_date = {December 2023},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        volume = {42},
        number = {6},
        issn = {0730-0301},
        url = {https://doi.org/10.1145/3618397},
        doi = {10.1145/3618397},
        journal = {ACM Trans. Graph.},
        month = {dec},
        articleno = {178},
        numpages = {16},
}
```

## Acknowledgements
This project is built upon the [Adversarial Skill Embeddings](https://github.com/nv-tlabs/ASE). We extend our gratitude to the ASE developers and contributors for laying the groundwork that has enabled the development of our repository. 

## License

Use MIT license (see LICENSE.txt) except for third-party softwares. They are all open-source softwares and have their own license types.
 
## Disclaimer
 
This is not an officially supported Tencent product. The code and data in this repository are for research purpose only. No representation or warranty whatsoever, expressed or implied, is made as to its accuracy, reliability or completeness. We assume no liability and are not responsible for any misuse or damage caused by the code and data. Your use of the code and data are subject to applicable laws and your use of them is at your own risk.