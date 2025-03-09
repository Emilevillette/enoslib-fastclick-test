import requests
# you may need to install requests for python3 with sudo-g5k apt install python3-requests
from statistics import mean
from time import sleep, time

def get_power(node, site, start, stop, metric="wattmetre_power_watt"):
    url = "https://api.grid5000.fr/stable/sites/%s/metrics?metrics=%s&nodes=%s&start_time=%s&end_time=%s" \
            % (site, metric, node, int(start), int(stop))
    data = requests.get(url, verify=False).json()
    return sum(item['value'] for item in data)/len(data)

import platform
h=platform.node().split(".")[0]
print(h)
p = get_power(h, "nancy", time()-4, time()-1, metric="pdu_outlet_power_watt")
print(f"RESULT-MAXWATT {p}")
