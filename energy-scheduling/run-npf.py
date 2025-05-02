import logging
import os
import time
from pathlib import Path
import enoslib as en
import re
from datetime import datetime


def replace_mac_addresses(lua_path, new_src_mac, new_dst_mac, output_path=None):
    """
    Replace srcmac and dstmac values in a Lua script with new MAC addresses.

    Args:
        lua_path (str): Path to the input Lua file.
        new_src_mac (str): New source MAC address (e.g., "aa:bb:cc:dd:ee:ff").
        new_dst_mac (str): New destination MAC address (e.g., "11:22:33:44:55:66").
        output_path (str, optional): Path to write the modified file.
                                     If None, overwrite the original file.
    """
    with open(lua_path, "r") as f:
        content = f.read()

    # Replace srcmac and dstmac with the new values
    content = re.sub(r'srcmac\s*=\s*"[0-9a-fA-F:]{17}"', f'srcmac = "{new_src_mac}"', content)
    content = re.sub(r'dstmac\s*=\s*"[0-9a-fA-F:]{17}"', f'dstmac = "{new_dst_mac}"', content)

    # Write to file
    out_path = output_path if output_path else lua_path
    with open(out_path, "w") as f:
        f.write(content)

    print(f"MAC addresses updated in '{out_path}'.")


# Import grid_reload_jobs_from_ids

en.init_logging(level=logging.INFO)
en.check()

# Import the utils module from the parent directory
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import utils
from AsynchronousLauncher import AsynchronousLauncher

CLUSTER = "gros"
SITE = "nancy"
USER = "evillett"
IMAGE_PATH = f"https://api.grid5000.fr/sid/sites/{SITE}/public/cdelzotti/img/dpdk-{CLUSTER}-nfs-nokey.yaml"
NPF_DEFAULT_FILE = "synced/test_dpdk.npf"

####################
# Argument parsing #
####################

NB_PAIRS = 1
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
        NB_PAIRS = int(sys.argv[i + 1])
    if arg == "--start-date":
        start_date = sys.argv[i + 1]
    if arg == "--hours":
        nb_hours = int(sys.argv[i + 1])
    if arg == "--split-variable":
        split_variable = sys.argv[i + 1]
    if arg == "--continue":
        experiment_id = sys.argv[i + 1]
        print(f"Continuing experiment {experiment_id}")
    if arg == "--npf-files":
        npf_dir = sys.argv[i + 1]
    if arg == "--job-id":
        job_id = sys.argv[i + 1]

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

private_net = en.G5kNetworkConf(type="kavlan-local", site=SITE, id="1+")

# nb_hours = 4

if job_id is None:
    reservation_base = en.G5kConf.from_settings(job_name="FastClick experiments",
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
    reservation_base = en.G5kConf.from_settings(job_name="FastClick experiments",
                                                job_type=["deploy"],
                                                env_name=f"https://api.grid5000.fr/sid/sites/{SITE}/public/cdelzotti/img/dpdk-{CLUSTER}-nfs-nokey.yaml",
                                                oargrid_jobids=[[SITE, job_id]],
                                                key="~/.ssh/id_ed25519.pub",
                                                dhcp=False).add_network_conf(private_net)

for i in range(NB_PAIRS):
    reservation_base.add_machine(roles=[f"client{i}", f"pair{i}"], cluster=CLUSTER, nodes=1
                                 , secondary_networks=[private_net]
                                 )
    reservation_base.add_machine(roles=[f"server{i}", f"pair{i}"], cluster=CLUSTER, nodes=1
                                 , secondary_networks=[private_net]
                                 )
conf = (reservation_base)

# Get actual resources
roles, networks, provider = utils.init_ressources(en, conf, must_bind=CLUSTER == "paravance", start_time=start_date)

print(roles)

####################
# Prepare Machines #
####################

# Sync local files to the nodes
print("Syncing files...")
for i in range(NB_PAIRS):
    client_hostname = list(roles[f"client{i}"].data)[0].address
    server_hostname = list(roles[f"server{i}"].data)[0].address
    os.system(
        f"scp -o StrictHostKeyChecking=no runAll.sh synced/read_g5k_power.py root@{server_hostname}.{SITE}_g5k:/root/energy-aware-packet-scheduling/")
    os.system(
        f"scp -o StrictHostKeyChecking=no runAll.sh synced/read_g5k_power.py root@{client_hostname}.{SITE}_g5k:/root/energy-aware-packet-scheduling/")

    os.system(f"scp -o StrictHostKeyChecking=no envs/env_{i}.sh root@{client_hostname}.{SITE}_g5k:/root/.env.sh")

# Keep a list of running commands
asyncLauncher = AsynchronousLauncher(en)
running_commands = []
server_hostnames = []
client_hostnames = []
fastclick_conf_opt = ["--enable-vector", "--enable-vector --disable-avx512", "--disable-avx512"]
for i in range(NB_PAIRS):
    os.system(f"scp -o StrictHostKeyChecking=no {npf_files[i]} root@{client_hostname}.{SITE}_g5k:/root/npf_script.npf")
    # Up the interface
    # Set interfaces IPs
    CLIENT_IP = f"192.168.192.{i * 2}/20"
    print(f"Client IP: {CLIENT_IP}")
    SERVER_IP = f"192.168.192.{i * 2 + 1}/20"
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
    server_hostnames.append(server_hostname)
    client_hostnames.append(client_hostname)
    # Get MAC addresses
    client_mac = en.run_command("ip link show eno2 | awk '/ether/ {print $2}'", roles=roles[f"client{i}"])[0].payload[
        "stdout"]
    server_mac = en.run_command("ip link show eno2 | awk '/ether/ {print $2}'", roles=roles[f"server{i}"])[0].payload[
        "stdout"]

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

    CLUSTER_DIR = f"./cluster"
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

    replace_mac_addresses("pktgen/cfg.lua", server_mac, client_mac)

    with open(f"{CLUSTER_DIR}/{server_hostname}.node", "w") as f:
        f.write(cluster_template)

    os.system(
        f"scp -o StrictHostKeyChecking=no {CLUSTER_DIR}/{client_hostname}.node root@{client_hostname}.{SITE}_g5k:/root/cluster/{client_hostname}.node")
    os.system(
        f"scp -o StrictHostKeyChecking=no {CLUSTER_DIR}/{server_hostname}.node root@{client_hostname}.{SITE}_g5k:/root/cluster/{server_hostname}.node")

    # Run experiment
    SOURCE_DATA_DIR = f"/root/energy-aware-packet-scheduling/results/f{SITE}/"
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

    print(client_hostname)
    print(server_hostname)

    os.system(
        f"rsync -avz --exclude '.git/' --exclude '.idea/'  -e 'ssh -o StrictHostKeyChecking=no' /home/emile/cours/UCLouvain/TFE/fastclick root@{client_hostname}.{SITE}_g5k:/root/")

    os.system(
        f"scp -o StrictHostKeyChecking=no pktgen/cfg.lua root@{server_hostname}.{SITE}_g5k:/root/cfg.lua")

    os.system(
        f"scp -o StrictHostKeyChecking=no synced/test_dpdk.npf root@{server_hostname}.{SITE}_g5k:/root/npf_script.npf"
    )

    os.system(
        f"scp -o StrictHostKeyChecking=no /home/emile/cours/UCLouvain/TFE/TFE-utils/enoslib/dpdk-23.03.tar.xz root@{server_hostname}.{SITE}_g5k:/root/"
    )

    os.system(
        f"scp -o StrictHostKeyChecking=no /home/emile/cours/UCLouvain/TFE/TFE-utils/enoslib/fastclick.sh root@{client_hostname}.{SITE}_g5k:/root/fastclick.sh"
    )

    # unpack dpdk
    # running_commands.append(
    #     asyncLauncher.run_command(
    #         f'tar -xf dpdk-23.03.tar.xz && rm dpdk-23.03.tar.xz',
    #         roles=roles[f"server{i}"], run_locally=False
    #     )
    # )

    # HERE /!\
    running_commands.append(
        asyncLauncher.run_command(
            f'apt install -y valgrind && /root/fastclick/configure CFLAGS="-msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -mfma -mbmi -mbmi2" CXXFLAGS="-msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -mfma -mbmi -mbmi2" --enable-dpdk --enable-bound-port-transfer --enable-flow --disable-task-stats --disable-cpu-load --enable-dpdk-packet --disable-clone --disable-dpdk-softqueue --disable-analysis --disable-app --disable-aqm --disable-simple --disable-tcpudp --disable-test --disable-threads --disable-flow --enable-dpdk-pool {fastclick_conf_opt[i]} && make -j 16 -C /root/fastclick install && tmux new -d -s fc "/usr/local/bin/click --dpdk -c 0x1 -n 4 -a 0000:18:00.1 -- /root/fastclick/conf/ip/decipttl.click; exec bash"',
            roles=roles[f"client{i}"], run_locally=False)
    )

    # running_commands.append(
    #     asyncLauncher.run_command(
    #         'apt install -y  liblua5.4-dev lua5.4 && cd /root/dpdk-23.03 && meson setup build -Dprefix=$(pwd)/install && export DPDK_PATH=$(pwd)/install && cd build && ninja && ninja install && export PKG_CONFIG_PATH=${DPDK_PATH}/lib/x86_64-linux-gnu/pkgconfig/ && export LD_LIBRARY_PATH=${DPDK_PATH}/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH && cd /root/ && rm -rf Pktgen-DPDK && git clone https://github.com/Emilevillette/Pktgen-DPDK.git && cd Pktgen-DPDK && git checkout working_lua && make buildlua',
    #         roles=roles[f"server{i}"], run_locally=False
    #     )
    # )

    # launch fastclick
    # running_commands.append(
    #     asyncLauncher.run_command(
    #         f"/usr/local/bin/click --dpdk -c 0x1 -n 4 -a 0000:18:00.1 -- /root/fastclick/conf/ip/decipttl.click",
    #         roles=roles[f"client{i}"], run_locally=False
    #     )
    # )

    # running_commands.append(
    #     asyncLauncher.run_command(
    #         "/root/fastclick.sh &; echo $! > /root/fastclick.pid",
    #         roles=roles[f"client{i}"], run_locally=False
    #     )
    # )

    # for j in range(len(running_commands)):
    #     running_commands[j].wait(polling_interval=15)
    #     print(f"LAUNCHING PKTGEN ON SERVER {i}")

    # time.sleep(15)
    # print(f"STARTING PKTGEN ON SERVER {i}")

    # Launch DPDK-pktgen
    # running_commands.append(
    #     asyncLauncher.run_command(
    #         f'/root/Pktgen-DPDK/usr/local/bin/pktgen -l 0-3 -n 4 -a 0000:18:00.1 -- -P -m "[1:2].0" -f /root/cfg.lua',
    #         roles=roles[f"server{i}"], run_locally=False
    #     )
    # )

    # running_commands.append(
    #     asyncLauncher.run_command(
    #         f'/root/npf/npf-run.py --test /root/npf_script.npf --single-output /root/results.csv',
    #         roles=roles[f"server{i}"], run_locally=False
    #     )
    # )

    # running_commands.append(
    #     asyncLauncher.run_command(
    #         'tar -xf dpdk-23.03.tar.xz && rm dpdk-23.03.tar.xz && chmod +x /root/npf_script.npf && apt install -y  liblua5.4-dev lua5.4 && cd /root/dpdk-23.03 && meson setup build -Dprefix=$(pwd)/install && export DPDK_PATH=$(pwd)/install && cd build && ninja && ninja install && export PKG_CONFIG_PATH=${DPDK_PATH}/lib/x86_64-linux-gnu/pkgconfig/ && export LD_LIBRARY_PATH=${DPDK_PATH}/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH && cd /root/ && rm -rf Pktgen-DPDK && git clone https://github.com/Emilevillette/Pktgen-DPDK.git && cd Pktgen-DPDK && git checkout working_lua && make buildlua && cd && /root/npf/npf-run.py --test /root/npf_script.npf --single-output /root/results.csv',
    #         roles=roles[f"server{i}"], run_locally=False
    #     )
    # )

    # Wait for the command to finish
    # running_commands[-1].wait(polling_interval=15)
    # print(f"PKTGEN ON SERVER {i} FINISHED, KILLING FASTCLICK")

for i in range(len(running_commands)):
    running_commands[i].wait(polling_interval=15)
    print(f"FASTCLICK INSTALL FINISHED ON CLIENT {client_hostnames[i]}")

# Launch pktgen
run_npf = False
for i in range(NB_PAIRS):
    if run_npf:
        running_commands.append(
            asyncLauncher.run_command(
                "tar -xf dpdk-23.03.tar.xz && rm dpdk-23.03.tar.xz && chmod +x /root/npf_script.npf && apt install -y  liblua5.4-dev lua5.4 && cd /root/dpdk-23.03 && meson setup build -Dprefix=$(pwd)/install && export DPDK_PATH=$(pwd)/install && cd build && ninja && ninja install && export PKG_CONFIG_PATH=${DPDK_PATH}/lib/x86_64-linux-gnu/pkgconfig/ && export LD_LIBRARY_PATH=${DPDK_PATH}/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH && cd /root/ && rm -rf Pktgen-DPDK && git clone https://github.com/Emilevillette/Pktgen-DPDK.git && cd Pktgen-DPDK && git checkout working_lua && make buildlua && cd && /root/npf/npf-run.py --test /root/npf_script.npf --single-output /root/results.csv",
                roles=roles[f"server{i}"], run_locally=False
            )
        )
####################
# Wait for results #
####################


for i in range(NB_PAIRS):
    print(f"CLIENT {i} : {client_hostnames[i]}")
    print(f"SERVER {i} : {server_hostnames[i]}")

# Wait for all the commands to finish
for i in range(len(running_commands)):
    running_commands[i].wait(polling_interval=15)
    print(f"Client {i} finished")

subname = ["VECTOR_AVX", "VECTOR_NOAVX", "LL_NOAVX"]
for i in range(NB_PAIRS):
    # get results.csv from server
    server_hostname = server_hostnames[i]
    os.system(
        f"scp -o StrictHostKeyChecking=no root@{server_hostname}.{SITE}_g5k:/root/results.csv ./results/results_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{subname[i]}.csv")
