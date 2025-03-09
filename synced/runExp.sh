#!/usr/bin/bash

cd /root/energy-aware-packet-scheduling/
source .env
npf-compare.py "fastclick-sleepmodes+SLEEP_MODE=no_sleep,SLOTH_ENABLED=0,dpdk:Polling" "fastclick-sleepmodes+SLEEP_MODE=no_sleep,SLOTH_ENABLED=1,dpdk:Sloth" "fastclick-sleepmodes+SLEEP_MODE=tupe,SLOTH_ENABLED=0:TUPE" --force-retest --test ./test_dpdk.npf --graph-filename ./results/$HOSTNAME/pmp-$HOSTNAME.pdf --tags dpdk trace timing iterative --variables LIMIT=1000000 FREQ=4000 PKTGEN_REPLAY_COUNT=100 PKTGEN_REPLAY_TIME=60 SAMPLE=100 --cluster client=$NODE1,nic=0 dut=$NODE2,nic=0 --graph-size 12 10 --single-output ./results/$HOSTNAME/pmp.csv --cluster-autosave --result-path ./results-$RATE/ --variables MINFREQ=1000  --show-cmd | tee ./results/$HOSTNAME/pmp-nosleep-$HOSTNAME.log
