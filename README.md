# EnosLib demo

This is a demo of how to use the EnosLib library to interact with Grid5000. This script reserves nodes on the Grid5000 *gros* cluster (by pair of 2 nodes) and deploys a simple experiment on them.

## Requirements

- Create a python venv and install enoslib and other dependencies 

You must setup your `.ssh/config` file in order to allow scripts to communicate with reserved nodes :

```
Host g5k
    ServerAliveInterval 60
    HostName access.grid5000.fr
    User G5K_USERNAME
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no

Host *.g5k
    User G5K_USERNAME
    ProxyCommand ssh g5k -W "$(basename %h .g5k):%p"
    ForwardAgent no

Host *.nancy_g5k
    User root
    ProxyCommand ssh nancy.g5k -W "$(basename %h .nancy_g5k):%p"
    ForwardAgent no
```

This shoud allow to reach any reserved node on Grid5000 from your local machine with such commands :

```bash
ssh gros-95.nancy.grid5000.fr.nancy_g5k # Opens a ssh connection to the node called gros-95
```

## First deployment

This script intends to facilitate the deployment of NPF experiments. Consequently, at runtime, you must fill either the `--npf-files` argument with a folder containing npf files to run for each pair, or the `--split-variable` argument that will fetch the default npf_file and generate a new one for each pair of nodes by spreading the values of the given variable. (e.g. if SLEEP_DELTA=[1-5] and 2 pairs are required, it will generate 2 npf files with SLEEP_DELTA={1,3,5} and SLEEP_DELTA={2,4}).

Although these NPF files are sent on each node during the setup, they won't be run automatically. When a pair is ready, it automatically launches the `runAll.sh` script. It's up to you to adapt it to your needs.

You can change what arguments this script receives in [`energy-scheduling/run-npf.py#238`](energy-scheduling/run-npf.py#238).

Once everything adapted, you can run the script with the following command :

```bash
# Reserve 2 pairs (4 nodes) on the gros cluster for 2 hours
python energy-scheduling/run-npf.py --pairs 2 --hours 2 --split-variable SLEEP_DELTA
```
## Storage

It's important to understand that reservations are limited in time and must comply with the [Usage Policy](https://www.grid5000.fr/w/Grid5000:UsagePolicy). Once your reservation is over, your machines are brutally reset and all your data is lost.

I advice you to check the [Grid5000 Storage Manager](https://www.grid5000.fr/w/Storage_Manager) page to understand how to share your frontend's home folder between all your nodes. This way, you can store your data on the frontend and retrieve it after the reservation is over.

By default, the provided image mounts my personal home folder to `/nfs`. You can change this by running `crontab -e` to adapt it to your needs. **You must have previously followed the [Grid5000 Storage Manager](https://www.grid5000.fr/w/Storage_Manager) instructions to grant access to your home folder**.

## Creating your own image

Once everything is customized, you can export it as an archive to redeploy it later. Just follow [this tutorial](https://www.grid5000.fr/w/Environment_creation#Archive_the_environment_image).

## Leveraging NPF with G5K

The NPF libary has a nice cache features that allows to restart experiments where they were stopped. Combined with the shared home folder, you can easily run your experiments on Grid5000 without losing any data and restarting them even after a brutal reset. Just make another reservation and restart your experiments.

To do so, you can provide an experiment id to the script :

```bash
python energy-scheduling/run-npf.py --pairs 8 --hours 4 --continue myniceexperiment --split-variable SLEEP_DELTA
```

If it's the first time you run this experiment, it will create a `myinitexperiment` folder run npf and store cache. If this folder already exists, it will load the cache and restart the experiment where it was stopped. Default experiments ids are ugly timestamps. You can also provide `--continue auto` to automatically fetch the last experiment id and continue it.

> You should launch the tests from your frontend, using `bash npf_script.bash`