import time
import logging
import os
from pathlib import Path
import enoslib as en
# Import grid_reload_jobs_from_ids
import pandas as pd
import sys

en.init_logging(level=logging.INFO)
en.check()

# Import the utils module from the parent directory
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import utils
from AsynchronousLauncher import AsynchronousLauncher

CLUSTER="parasilo"
SITE="rennes"
USER="evillett"
IMAGE_PATH="https://api.grid5000.fr/sid/sites/{SITE}/public/{USER}/img/dpdk-{CLUSTER}-nfs-nokey.yaml"
NPF_DEFAULT_FILE = "synced/test_dpdk.npf"

####################
# Argument parsing #
####################

NB_PAIRS=1
start_date = None
experiment_id = None
npf_dir = ""
job_id = None

if "--help" in sys.argv:
    print("""
          --pairs <int> : Number of pairs to run
          --start-date <str> : Start date of the reservation
          --hours <int> : Number of hours to run the experiment
          --split-variable <str> : Variable to split the results on
          --continue <experiment id> : Continue a already started experiment
          --npf-files <path> : Path to the npf files
          --job-id <str> : Job id to reload
          """)
    sys.exit(0)

for i, arg in enumerate(sys.argv):
    if arg == "--pairs":
        NB_PAIRS = int(sys.argv[i+1])
    if arg == "--start-date":
        start_date = sys.argv[i+1]
    if arg == "--hours":
        nb_hours = int(sys.argv[i+1])
    if arg == "--split-variable":
        split_variable = sys.argv[i+1]
    if arg == "--continue":
        experiment_id = sys.argv[i+1]
        print(f"Continuing experiment {experiment_id}")
    if arg == "--npf-files":
        npf_dir = sys.argv[i+1]
    if arg == "--job-id":
        job_id = sys.argv[i+1]

##################################
# Generate or retrieve NPF files #
##################################

if npf_dir == "":
    print(f"Splitting on {split_variable} with {NB_PAIRS} pairs")
    npf_files = utils.npf_split(NPF_DEFAULT_FILE, split_variable, NB_PAIRS)
else:
    print(f"Using provided npf files within {npf_dir}")
    # Get a list of the files in the directory
    npf_files = os.listdir(npf_dir)
    # Add the directory to the files
    npf_files = [f"{npf_dir}/{f}" for f in npf_files]
    # Order in alphabetical order
    npf_files.sort()
print(f"Split files: {npf_files}")

########################
# Define experiment ID #
########################

# If no experiment id was provided, generate one
if experiment_id is None:
    experiment_id = f"{utils.get_timestamp()}"
    with open("experiment_ids.donotcommit", "a+") as f:
        # Write the experiment id to the history file
        f.write(f"{experiment_id}\n")

# If the experiment id is "auto", get the last experiment id
if experiment_id == "auto":
    with open("experiment_ids.donotcommit", "r") as f:
        lines = f.readlines()
        experiment_id = lines[-1].strip()

##################
# Get resources  #
##################

private_net = en.G5kNetworkConf(type="kavlan", roles=["private"], site=SITE,id="1+") 

    # nb_hours = 4

if job_id is None:
    reservation_base = en.G5kConf.from_settings(job_name="testing stuff",
                                job_type=["deploy"],
                                env_name=f"https://api.grid5000.fr/sid/sites/{SITE}/public/cdelzotti/img/dpdk-{CLUSTER}-nfs-nokey.yaml",
                                # Make the reservation last 9 hours
                                walltime=f"{nb_hours}:00:00",
                                # Only 1 switch
                                # switch=1,
                                key="~/.ssh/id_ed25519.pub",
                                dhcp=False).add_network_conf(private_net)
else:
    print(f"Reloading job {job_id}")
    reservation_base = en.G5kConf.from_settings(job_name="testing stuff",
                                job_type=["deploy"],
                                env_name=f"https://api.grid5000.fr/sid/sites/{SITE}/public/cdelzotti/img/dpdk-{CLUSTER}-nfs-nokey.yaml",
                                oargrid_jobids=[[SITE, job_id]],
                                key="~/.ssh/id_ed25519.pub",
                                dhcp=False).add_network_conf(private_net)

for i in range(NB_PAIRS):
    reservation_base.add_machine(roles=[f"client{i}", f"pair{i}"], cluster=CLUSTER, nodes=1
                                ,secondary_networks=[private_net]
                                )
    reservation_base.add_machine(roles=[f"server{i}", f"pair{i}"], cluster=CLUSTER, nodes=1
                                ,secondary_networks=[private_net]
                                )
conf = (reservation_base)

# Get actual resources
roles, networks, provider = utils.init_ressources(en, conf, must_bind=CLUSTER=="parasilo", start_time=start_date)

####################
# Prepare Machines #
####################

# Sync local files to the nodes
print(("Syncing files..."))
for i in range(NB_PAIRS):
    client_hostname = list(roles[f"client{i}"].data)[0].address
    server_hostname = list(roles[f"server{i}"].data)[0].address
    os.system(f"scp -o StrictHostKeyChecking=no runAll.sh synced/read_g5k_power.py root@{server_hostname}.{SITE}_g5k:/root/energy-aware-packet-scheduling/")
    os.system(f"scp -o StrictHostKeyChecking=no runAll.sh synced/read_g5k_power.py root@{client_hostname}.{SITE}_g5k:/root/energy-aware-packet-scheduling/")
    
# Keep a list of running commands
asyncLauncher = AsynchronousLauncher(en)
running_commands = []
for i in range(NB_PAIRS):
    os.system(f"scp -o StrictHostKeyChecking=no {npf_files[i]} root@{client_hostname}.{SITE}_g5k:/root/npf_script.npf")
    # Up the interface
    # Set interfaces IPs
    CLIENT_IP=f"192.168.192.{i*2}/20"
    print(f"Client IP: {CLIENT_IP}")
    SERVER_IP=f"192.168.192.{i*2+1}/20"
    print(f"Server IP: {SERVER_IP}")
    results = en.run_command("ip link set eno2 up", roles=roles[f"pair{i}"])
    try:
        results = en.run_command(f"ip addr add {CLIENT_IP} dev eno2", roles=roles[f"client{i}"])
        results = en.run_command(f"ip addr add {SERVER_IP} dev eno2", roles=roles[f"server{i}"])
    except:
        # Setting IP fails if the interface is already set up. In this case just ignore the error
        pass

    # Retrieve hostnames
    client_hostname = list(roles[f"client{i}"].data)[0].address
    server_hostname = list(roles[f"server{i}"].data)[0].address
    # Get MAC addresses
    client_mac = en.run_command("ip link show eno2 | awk '/ether/ {print $2}'", roles=roles[f"client{i}"])[0].payload["stdout"]
    server_mac = en.run_command("ip link show eno2 | awk '/ether/ {print $2}'", roles=roles[f"server{i}"])[0].payload["stdout"]


    # Load Metronome kernel module
    try:
        results = en.run_command("insmod /root/Metronome/hr_sleep/hr_sleep.ko", roles=roles[f"pair{i}"])
    except:
        # If the module is already loaded, ignore the error
        pass

    # Install unbuffer
    results = en.run_command("apt update && apt install -y expect", roles=roles[f"pair{i}"])

    # Set client environment
    try:
        # Remove the file if it already exists
        results = en.run_command("rm /root/energy-aware-packet-scheduling/.env", roles=roles[f"pair{i}"])
    except:
        pass

    CLUSTER_DIR=f"./cluster"
    NODE_TEMPLATE = f"{CLUSTER_DIR}/node-{CLUSTER}.node"
    
    # Open template and read the content
    with open(NODE_TEMPLATE, "r") as f:
        cluster_template = f.read()
        
    # Set client cluster
        
    # Add customized lines
    cluster_template += f"0:mac={client_mac}\n"
    cluster_template += f"0:ip={CLIENT_IP[:-3]}\n"
    # Write the new content to the file
    with open(f"{CLUSTER_DIR}/{client_hostname}.node", "w") as f:
        f.write(cluster_template)
        
    # Set server cluster
    cluster_template = cluster_template.replace(client_mac, server_mac)
    cluster_template = cluster_template.replace(CLIENT_IP[:-3], SERVER_IP[:-3])
    
    with open(f"{CLUSTER_DIR}/{server_hostname}.node", "w") as f:
        f.write(cluster_template)
    
    os.system(f"scp -o StrictHostKeyChecking=no {CLUSTER_DIR}/{client_hostname}.node root@{client_hostname}.{SITE}_g5k:/root/cluster/{client_hostname}.node")
    os.system(f"scp -o StrictHostKeyChecking=no {CLUSTER_DIR}/{server_hostname}.node root@{client_hostname}.{SITE}_g5k:/root/cluster/{server_hostname}.node")
    
    # Run experiment
    SOURCE_DATA_DIR="/root/energy-aware-packet-scheduling/results/f{SITE}/"
    # Clean the results directory
    en.run_command(f"rm -rf {SOURCE_DATA_DIR}/*", roles=roles[f"pair{i}"])
    # Create local environment directory
    os.makedirs(f"envs", exist_ok=True)
    
    # Create the environment file
    with open(f"envs/env_{i}.sh", "w") as f:
        f.write(f"export CLIENT={client_hostname}\n")
        f.write(f"export SERVER={server_hostname}\n")
        f.write(f"export PATH=/root/npf:$PATH\n")
        f.write(f"export HOSTNAME=f{SITE}\n")
        f.write(f"export INsrcmac={client_mac}\n")
        f.write(f"export INdstmac={server_mac}\n")
        f.write("export NODE_MIN_FREQ=1000\n")
        f.write("export SAMPLE=10\n")
        f.write("export LIMIT=500000000\n")
    
    os.system(f"scp -o StrictHostKeyChecking=no envs/env_{i}.sh root@{client_hostname}.{SITE}_g5k:/root/.env.sh")
    
    running_commands.append(
        asyncLauncher.run_command(f"/root/energy-aware-packet-scheduling/runAll.sh g5k ~/.env.sh /root/npf_script.npf {experiment_id} {i} | tee /nfs/log_{i}.donotcommit", roles=roles[f"client{i}"], run_locally=False)
    )
    
####################
# Wait for results #
####################
    
# Wait for all the commands to finish
for i in range(len(running_commands)):
    running_commands[i].wait(polling_interval=15)
    print(f"Client {i} finished")