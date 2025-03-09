import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import seaborn as sns

def avg(l):
    """
    Returns the average of a list.

    Parameters
    ----------
    l : List
        The list to average

    Returns
    -------
    avg : float
        The average of the list
    """
    return sum(l) / len(l)

def middle(l):
    """
    Returns the middle of a list.

    Parameters
    ----------
    l : List
        The list to get the middle of

    Returns
    -------
    middle : float
        The middle of the list
    """
    return l[len(l) // 2]

def build_lat_graph(data, save_path="./graph-lat.pdf",maxlat=120, metric='LAT99', core_axis=True):
    """
    Generates a graph of the best configurations (WATT, Freq, Cores, Sleep mode) for each latency objective.
    
    Parameters
    ----------
    data : DataFrame
        The data to plot
    save_path : String
        The path to save the graph
    maxlat : int
        The maximum latency objective to plot. The interval will be [Smallest possible with data, maxlat]
    metric : String
        The metric to plot (LAT99, LAT50, etc.)
    core_axis : bool
        Whether to plot the number of cores on the second y axis with a bar plot behind the points
    """
    #best_nosleep = get_best_power(data[(data["SLEEP_MODE"] == "no_sleep") & (data["FREQ"] == freqs[-1])])
    #print("Best without sleep : ")
    #print(best_nosleep)
    #lat_target = best_nosleep["LAT99"].squeeze()
    #print("LAT99:",  lat_target)
    colors = {}
    sleep_modes = data["SLEEP_MODE"].unique()
    for i, sleep_mode in enumerate(sleep_modes):
        colors[sleep_mode] = sns.color_palette("tab10")[i]
    plt.clf()
    
    lats = np.array(range(1,maxlat))
    lines=dict()
    lastbest=None
    color_grad=[]
    min_lat = 0
    for l in lats:
        bdata = data[data["LAT50"] < l]
        min_lat = int(data["LAT50"].min())
        #print("Valid sleep modes : ", bdata['SLEEP_MODE'].unique(), len(bdata))
        best=bdata.sort_values("WATT")[:1]
        if len(best) == 0:
            #print(f"No value for {l}")
            continue
        freq_color=(best["FREQ"].to_list()[0]/3000)
        #v.append(best[CPU])
        #display(best)
        #print(int(best['CPU'].squeeze()), int(best['FREQ'].squeeze()),best['WATT'].squeeze(), best['SLEEP_MODE'].squeeze() )
        watt=best['WATT'].squeeze()
        lines.setdefault(f"{best['SLEEP_MODE'].squeeze()} - {int(best['CPU'].squeeze())} cores - {best['FREQ'].squeeze()} MHz",[]).append(tuple([l,watt,best["FREQ"].to_list()[0]/3900]))
        plt.plot(int(l), watt, "o")
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(12,6))
    # Set x label
    # Add a twinx to show the number of cores in a bar plot

    # Set y ticks size for both y axis
    # Do the same for x axis
    if core_axis:
        ax2 = ax1.twinx()
        ax1.set_xlabel("Median latency objective (µs)", fontsize=15)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)
    else:
        ax2 = ax1
    ax2.set_xlabel("Median latency objective (µs)", fontsize=15)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='x', labelsize=12)
    # ax1.set_zorder(1)
    i = 0
    legend = {}
    cores = []
    sleep_modes = []
    for l,p in lines.items():
        xy = list(zip(*p))
        # print(xy)
        labels=str(l).split(" - ")
        nb_cores=int(labels[1].split(" ")[0])
        cores += [nb_cores]*len(xy[0])
        sleep_modes += [labels[0]]*len(xy[0])
        ax2.plot(xy[0], xy[1], "o", color=plt.cm.viridis(avg(xy[2])))
        # Draw a box around the points
        # plt.plot([middle(xy[0]), middle(xy[0]), middle(xy[0]), middle(xy[0]), middle(xy[0])], [min(xy[1])-nb_cores/2, min(xy[1])-nb_cores/2, max(xy[1])+nb_cores/2, max(xy[1])+nb_cores/2, min(xy[1])-nb_cores/2], color='darkcyan')
        
        additional_shift_x=0
        if nb_cores > 9:
            additional_shift_x=-1
            if len(xy[0]) > 1:
                additional_shift_x+=0.5
        # Add 1 in list xy[0] at index 0
        xy[0] = [xy[0][0]-0.5] + list(xy[0]) + [xy[0][-1]+0.5]
         
        # additional_shift_x=0
        # if len(xy[0]) % 2 != 0 and len(xy[0]) != 1:
        #     additional_shift_x=-0.5

        # additional_shift_y = 0
        # if len(xy[0]) < 3:
        #     additional_shift_y = -1
        # plt.fill_between(xy[0], min(xy[1])-10, max(xy[1])+10, color=human_to_ensg_dictionary[labels[0]], alpha=0.4)
        legend[labels[0]] = colors[labels[0]]
        # plt.text(middle(xy[0]) -0.5+additional_shift_x, middle(xy[1]) + 4*core_position , f'{nb_cores}', color="black", fontsize=12)


        # if i %2 == 0:
        i += 1
        plt.legend(fontsize=14, framealpha=0.5,loc="upper right",ncol=2)
    # Plot 'cores' variable as a bar plot on the second y axis
    x_ticks = np.arange(min_lat + 1, min_lat + len(cores) + 1)
    colors = [colors[s] for s in sleep_modes]
    if core_axis:
        ax1.bar(x_ticks, cores, color=colors, alpha=0.4)
        ax1.set_ylabel("Number of cores", fontsize=15)
        # Hide grid for second ax
        ax1.grid(False)
    ax2.set_ylabel("Power (W)", fontsize=15)
    ax2.grid(True)
    # Use `legend` dictionary to manually set color -> label legend
    plt.legend(handles=[plt.Line2D([0], [0], color=value, label=key) for key, value in legend.items()], fontsize=12, framealpha=0.4,loc="upper right",ncol=2)
    # Force Y axis to start at 0
    plt.ylim(bottom=0)
    # Ensure no color overlap
    plt.tight_layout()
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=plt.gca(), location="right")
    # Shift colorbar 2cm to the right
    cbar.ax.set_position([0.85, 0.15, 0.05, 0.7])
    # Set label fontsize
    cbar.set_label('% of maximal frequency', fontsize=15)
    # Set ticks fontsize
    cbar.ax.tick_params(labelsize=12)
    # Save the figure as pdf
    plt.savefig(save_path)
    plt.show()
    plt.clf()
    
def build_lat_gain_graph(data1, data2, save_path="./graph-gain-lat.pdf",maxlat=120, metric='LAT50'):
    """
    Generates a graph of the best configurations (WATT, Freq, Cores, Sleep mode) for each latency objective.
    
    Parameters
    ----------
    data : DataFrame
        The data to plot
    save_path : String
        The path to save the graph
    maxlat : int
        The maximum latency objective to plot. The interval will be [Smallest possible with data, maxlat]
    metric : String
        The metric to plot (LAT99, LAT50, etc.)
    core_axis : bool
        Whether to plot the number of cores on the second y axis with a bar plot behind the points
    """
    #best_nosleep = get_best_power(data[(data["SLEEP_MODE"] == "no_sleep") & (data["FREQ"] == freqs[-1])])
    #print("Best without sleep : ")
    #print(best_nosleep)
    #lat_target = best_nosleep["LAT99"].squeeze()
    #print("LAT99:",  lat_target)
    plt.clf()
    data1_watt=[]
    data2_watt=[]
    for l in np.array(range(1,maxlat)):
        bdata1 = data1[data1["LAT50"] < l]
        best1=bdata1.sort_values("WATT")[:1]
        
        bdata2 = data2[data2["LAT50"] < l]
        best2=bdata2.sort_values("WATT")[:1]
        
        if len(best1) == 0 or len(best2) == 0:
            continue
        data1_watt.append(best1['WATT'].squeeze())
        print(data1_watt)
        data2_watt.append(best2['WATT'].squeeze())
    plt.plot(data1_watt, label="Optimal")
    plt.plot(data2_watt, label="DVFS only")
    # Set x label
    plt.xlabel("Median latency objective (µs)", fontsize=15)
    # Set y label
    plt.ylabel("Power gain (W)", fontsize=15)
    # Set ticks size
    plt.xticks(fontsize=1)
    plt.yticks(fontsize=12)
    # Show grid
    plt.grid(True)
    # Force y axis to start at 0
    plt.ylim(bottom=0)
    # Show legend
    plt.legend(fontsize=12)
    # Save the figure as pdf
    plt.savefig(save_path)
    plt.show()
    plt.clf()
    
def build_cpu_graph(data,y_col='WATT',div=1.0,agg='min', save_path="./graph-cpu.pdf"):
    
    cpus = np.unique(data["CPU"])

    afreqs = np.unique(data['FREQ'])
    d = np.empty((len(cpus),len(afreqs)))
    max_watt = 0
    min_watt = 1e9
    for i_cpu,n_cpu in enumerate(cpus):
        for i_freq,freq in enumerate(afreqs):
            da = data[(data["CPU"] == n_cpu) & (data["FREQ"] == freq)]
            d[i_cpu,i_freq] = da[y_col].aggregate(agg) / div
            if d[i_cpu,i_freq] > max_watt:
                max_watt = d[i_cpu,i_freq]
            if d[i_cpu,i_freq] < min_watt:
                min_watt = d[i_cpu,i_freq]
    if d.size == 0:
        return
    plt.clf()
    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(d)
    ax.set_yticks(np.arange(len(cpus)), labels=cpus)
    ax.set_xticks(np.arange(len(afreqs)), labels=[int(f) for f in afreqs])
    
    print(f"Max : {max_watt} Min : {min_watt}")
    im.set_clim(min_watt, max_watt + 10)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(ylabel=y_col, rotation=-90, va="bottom")
    m = np.max(d)
    mi = np.min(d)
    r = 1
    if m < 10:
        r = 0

    for i in range(len(cpus)):
        for j in range(len(afreqs)):
            text = ax.text(j, i, (round(d[i, j], r) if d[i, j] < 10 else round(d[i, j])) if not np.isnan(d[i, j]) else "",
                       ha="center", va="center", color="black" if d[i, j] <= (mi+2*((m-mi)/4)) else "w")
    ax.set_title(f"{y_col} per CPU and frequency", fontsize=18)
    # Set x and y labels
    ax.set_xlabel("Frequency (MHz)",fontsize=14)
    ax.set_ylabel("Number of CPUs", fontsize=14)
    # Change ticks size
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    # Hide grid
    ax.grid(False)
    # Save the figure as pdf
    plt.savefig(save_path)
    
    plt.show()
    plt.clf()


def reboot_nodes(en, roles, provider):
    """
    Reboots the nodes.
    Parameters
    ----------
    en : enoslib
        The enoslib library
    roles : Dict
        The roles of the nodes
    provider : en.G5k
        The provider object
    Returns
    -------
    roles : Dict
        The roles of the nodes
    networks : Dict
        The networks of the nodes
    """
    try:
        en.run_command("reboot", roles=roles)
    except en.errors.EnosUnreachableHostsError as e:
        timestamp = time.strftime("%H%M%S")
        print(f"{timestamp} : Connection lost, waiting for nodes to be rebooted...")
        time.sleep(300)
        # Now reconnect to the nodes without triggering deployment
        roles, networks = provider.init()
        roles = en.sync_info(roles, networks)
        return roles, networks

def init_ressources(en, conf, must_bind=False, start_time=None):
    """
    Acquires resources and performs post-reservation operations to set up hugepages
    and MST.
    Parameters
    ----------
    en : enoslib
        The enoslib library
    conf : en.G5kConf
        The configuration object
    must_bind : bool (default : False)
        Whether the nodes must be binded (With Intel NICs)
    start_time : String (default : run as soon as possible)
        The start time of the reservation (format : "YYYY-MM-DD HH:MM:SS")
    Returns
    -------
    roles : Dict
        The roles of the nodes
    networks : Dict
        The networks of the nodes
    provider : en.G5k
        The provider object
    """
    # Acquire resources
    provider = en.G5k(conf)
    print("Waiting for nodes to be ready...")
    if start_time is None:
        roles, networks = provider.init()
    else:
        timestamp = time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
        roles, networks = provider.init(start_time=timestamp)
    roles = en.sync_info(roles, networks)
    print("Ressorces acquired, performing post-reservation operations...")
    # Check hugepages
    results = en.run_command("cat /proc/meminfo | grep HugePages_Total | tr \" \" \"\n\" | tail -n 1", roles=roles)
    hugepages_ready = True
    for result in results:
        if result.payload["stdout"] == "0":
            print(f"{result.host} : No hugepages available. Please reboot.")
            hugepages_ready = False
        else:
            print(f"{result.host} : {result.payload['stdout']} hugepages available.")
    # Reboot the nodes with enoslib
    if not hugepages_ready:
        # Update grub
        en.run_command("update-grub && update-grub2", roles=roles)
        print("Rebooting nodes...")
        roles, networks = reboot_nodes(en, roles, provider)
        print("Nodes rebooted.")
    # Initialize MST
    results = en.run_command("modprobe uio_pci_generic && mst start", roles=roles)
    # Set sysctl vm.stat_interval=120
    en.run_command("sysctl -w vm.stat_interval=120", roles=roles)
    # Set irqbalance
    en.run_command("irqbalance --foreground --oneshot", roles=roles)
    # Disable swap
    en.run_command("swapoff -a", roles=roles)
    # Disable THP
    en.run_command("echo never >| /sys/kernel/mm/transparent_hugepage/enabled", roles=roles)
    # Disable KSM
    en.run_command("echo 0 >| /sys/kernel/mm/ksm/run", roles=roles)
    # Paravance cluster needs to be binded
    if must_bind:
        try:
            # Load kernel module
            en.run_command("modprobe vfio-pci", roles=roles)
            en.run_command("modprobe vfio enable_unsafe_noiommu_mode=1", roles=roles)
            en.run_command("echo 1 > /sys/module/vfio/parameters/enable_unsafe_noiommu_mode", roles=roles)
            en.run_command("dpdk-devbind.py -u 0000:01:00.1", roles=roles)
            # Bind it with DPDK driver
            en.run_command("dpdk-devbind.py -b vfio-pci 0000:01:00.1", roles=roles)
            # Unbind port
            en.run_command("ip link set dev eno2 down", roles=roles)
        except:
            # Commands will fail if interface is already down
            pass
    return roles, networks, provider


def run_command(command, hosts):
    for host in hosts:
        try:
            os.system(f"ssh -o StrictHostKeyChecking=no root@{host}.nancy_g5k '{command}'")
        except:
            if "reboot" not in command:
                print(f"Error while running the command on {host}")
            else:
                print(f"Rebooting {host}, connection will be lost")


class creditentials:
    """
    This class is used to store the creditentials of the user.
    """
    def __init__(self, user, password):
        self.user = user
        self.password = password

    def get_user(self):
        return self.user

    def get_password(self):
        return self.password

    def get_tuple(self):
        return (self.user, self.password)

def get_env(variable_name):
    """
    Returns the value of the environment variable `variable_name`. If not set,
    it raises an exception.

    Returns
    -------
    variable_value : String
    """
    variable_value = os.environ.get(variable_name)
    if variable_value is None:
        raise Exception(f"Please set the environment variable {variable_name}")
    return variable_value

def get_creditentials():
    """
    To avoid any GitHub copilot snitching, this function only retrieves the
    content from environment variables. It checks if the variables `G5K_USER` and
    `G5K_PASS` are set, and returns them if they are.
    Returns
    -------
    CREDITENTIALS : Object containing the user and password
    """
    return creditentials(get_env("G5K_USER"), get_env("G5K_PASS"))


def get_timestamp():
    """
    Returns the current timestamp in the format YYYYMMDDHHMMSS
    Returns
    -------
    timestamp : String
    """
    return time.strftime("%Y%m%d%H%M%S")

def replace_line_in_config(line, new_line, en, roles):
    """
    Replaces a line in the config file with a new line using enoslib.
    Parameters
    ----------
    line : String
        The line to replace
    new_line : String
        The new line
    en : enoslib
        The enoslib library
    roles : Dict
        The roles of the nodes to run the command on
    Returns
    -------
    None
    """
    result = en.run_command(f"cd /root/energy-aware-packet-scheduling/ && cat test_dpdk.npf | grep '{line}' | head -n 1", roles=roles)
    search_string = result[0].payload['stdout']
    en.run_command(f"cd /root/energy-aware-packet-scheduling/ && cat test_dpdk.npf |  head -n $( echo $(grep -B 200 '{search_string}' test_dpdk.npf | wc -l) - 1 | bc) > test_dpdk2.npf && echo '{new_line}' >> test_dpdk2.npf && cat test_dpdk.npf | tail -n $( echo $(cat test_dpdk.npf | wc -l) - $(grep -B 200 '{search_string}' test_dpdk.npf | wc -l) | bc) >> test_dpdk2.npf && rm test_dpdk.npf && mv test_dpdk2.npf test_dpdk.npf", roles=roles)

def parition_config(config, nb_pairs):
    """
    Partition the configuration into multiple configurations to be run on different pairs

    Parameters
    ----------
    config : Dict
        The configuration to partition
    nb_pairs : int
        The number of pairs to partition the configuration into

    Returns
    -------
    partitioned_config : List
        The partitioned configuration
    """
    # Find the most provided parameter
    most_provided = max(config, key=lambda x: len(config[x]) if x != "NB_RUN" else 0)
    # Check that there are enough elements to provide all the pairs
    if len(config[most_provided]) < nb_pairs:
        print(f"Warning: most provided parameter {most_provided} has only {len(config[most_provided])} elements, while {nb_pairs} are required.")
        print(f"{nb_pairs - len(config[most_provided])} pairs will be missing.")
    # Prepare the partitioned config
    partitioned_config = []
    for i in range(min(nb_pairs, len(config[most_provided]))):
        partitioned_config.append(config.copy())
        partitioned_config[-1][most_provided] = []
    # Partition the configuration
    for i in range(len(config[most_provided])):
        partitioned_config[i % nb_pairs][most_provided].append(config[most_provided][i])
    return partitioned_config

def npf_split(file_path, variable_split, nb_split, output_dir="./npf-splits/"):
    # Open the file
    with open(file_path, 'r') as file:
        # Read the file
        data = file.readlines()
    
    # Go through the file
    matches = []
    for i, line in enumerate(data):
        # If the line describes the variable to split
        if line.startswith(f"{variable_split}=") or f":{variable_split}=" in line:
            # Get tags
            tags = ""
            if ":" in line:
                tags = line.split(":")[0]
            # Parse the value
            value = line.split("=")[1].strip()
            if "{" in value:
                value = value.split("{")[1].split("}")[0].split(",")
            elif "[" in value:
                step = 1
                start, end = value.split("[")[1].split("]")[0].split("-")
                if "#" in end:
                    end, step = end.split("#")
                value = [str(val) for val in list(range(int(start), int(end)+1, int(step)))]
            else:
                # Splitting a single value is not possible
                continue
            splits = [[] for _ in range(nb_split)]
            for j in range(max(len(value), nb_split)):
                splits[j % nb_split].append(value[j % len(value)])
            matches.append((i, splits, tags))
        
    # Check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Clear the directory
        for file in os.listdir(output_dir):
            os.remove(f"{output_dir}/{file}")
    # Generate the new files
    files = []
    for i in range(nb_split):
        new_data = data.copy()
        for j, split, tags in matches:
            # Build new line
            start=""
            if tags != "":
                start = f"{tags}:"
            new_data[j] = f"{start}{variable_split}={{{','.join(split[i])}}}\n"
        files.append(f"{output_dir}/test_dpdk_{i}.npf")
        with open(files[-1], 'w') as file:
            file.writelines(new_data)
    return files