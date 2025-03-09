#!/usr/bin/bash

# Ensure module is loaded


echo $(which npf-compare.py)

if [[ $1 == *"g5k"* ]]; then
    echo "We're on Grid5000, load from provided env"
    # Get first parameter
    source /root/energy-aware-packet-scheduling/venv/bin/activate
    source $2
    SETUP=g5k
    SCRIPT_PATH=$3
    EXPERIMENT_ID=$4
    PAIR_ID=$5
    # Setup working directory
    MAIN_DIR=/nfs/experiments_results/$EXPERIMENT_ID/$PAIR_ID
    mkdir -p $MAIN_DIR
    cd $MAIN_DIR
    rm -rf ./cluster
    cp -r ~/cluster cluster
    cp -r ~/repo repo
    export PATH=/root/npf2:$PATH
    MAX_THROUGHPUT=28000000
    MIN_FREQ=800
    MAX_FREQ=3000
    MAX_CORES=16
else
    echo "We're not on Grid5000, load from variables"
    CLIENT=merry
    SERVER=galadriel
    SETUP=galadriel
    SCRIPT_PATH=./synced/test_dpdk.npf
    MAIN_DIR=./results/$HOSTNAME-nat/
    export PATH=/etinfo/users2/delzotti/npf2:$PATH
    MAX_THROUGHPUT=28000000
    MIN_FREQ=1000
    MAX_FREQ=3900
    MAX_CORES=10
fi

# First plot : Compare DPDK vs Socket
# echo "Experiment 1 : Comparing DPDK vs Socket"
# RESULT_DIR=$MAIN_DIR/sockets-vs-dpdk/
# npf-compare.py "fastclick-sleepmodes+SLEEP_MODE=no_sleep,SLOTH_ENABLED=0,dpdk:DPDK" "fastclick-sleepmodes+SLEEP_MODE=no_sleep,SLOTH_ENABLED=0,socket:Socket"  --test $SCRIPT_PATH --graph-filename $RESULT_DIR/pmp-$HOSTNAME.pdf --tags socketdpdk dpdk $SETUP udpgen prate promisc --variables DISTRIB=EXP LIMIT=100000000 PKTGEN_REPLAY_COUNT=100 PKTGEN_REPLAY_TIME=60 SAMPLE=100 --cluster client=$CLIENT,nic=0 dut=$SERVER,nic=0 --graph-size 12 10 --single-output $RESULT_DIR/pmp.csv --cluster-autosave --result-path $RESULT_DIR/results --variables SLO_LATENCY=50 MINFREQ=$MIN_FREQ SLEEP_DELTA=1 --config n_runs=5 | tee $RESULT_DIR/pmp-nosleep-$HOSTNAME.log
# npf-compare.py "fastclick-sleepmodes+SLEEP_MODE=no_sleep,SLOTH_ENABLED=0,dpdk:DPDK" --test $SCRIPT_PATH --graph-filename $RESULT_DIR/pmp-$HOSTNAME.pdf --tags socketdpdk dpdk $SETUP udpgen prate promisc --variables DISTRIB=EXP LIMIT=100000000 PKTGEN_REPLAY_COUNT=100 PKTGEN_REPLAY_TIME=60 SAMPLE=100 --cluster client=$CLIENT,nic=0 dut=$SERVER,nic=0 --graph-size 12 10 --single-output $RESULT_DIR/pmp.csv --cluster-autosave --result-path $RESULT_DIR/results --variables SLO_LATENCY=50 MINFREQ=$MIN_FREQ SLEEP_DELTA=1 --config n_runs=5 | tee $RESULT_DIR/pmp-nosleep-$HOSTNAME.log

# # Second plot : just use experiment 1 results with only no_sleep and without losses

# # Third plots : build heatmap from cores and frequency

#Fourth plots : Sleep modes comparison
echo "Experiment 2 : Sleep modes"
RESULT_DIR=$MAIN_DIR/sleep-modes/
npf-compare.py "fastclick-sleepmodes+SLEEP_MODE=no_sleep,SLOTH_ENABLED=0,dpdk:Polling" "fastclick-sleepmodes+SLEEP_MODE=hr2sleep_add_simple,SLOTH_ENABLED=0,dpdk:Progressive HR-sleeping" "fastclick-sleepmodes+SLEEP_MODE=hr2sleep_constant_simple,SLOTH_ENABLED=0,dpdk:Constant HR-sleeping" "fastclick-sleepmodes+SLEEP_MODE=hr2sleep_constant_power_intr,SLOTH_ENABLED=0,dpdk:Constant HR-sleeping + interrupt" "fastclick-sleepmodes+SLEEP_MODE=hr2sleep_add_simple_intr,SLOTH_ENABLED=0,dpdk:Progressive HR-sleeping + interrupt" "fastclick-sleepmodes+SLEEP_MODE=add_simple_intr,SLOTH_ENABLED=0,dpdk:Progressive nanosleeping + interrupt" --test $SCRIPT_PATH --graph-filename $RESULT_DIR/pmp-$HOSTNAME.pdf --tags sleep dpdk $SETUP udpgen prate promisc --variables DISTRIB=EXP SLO_LATENCY=50 LIMIT=50000000 SLO_LATENCY=50 PKTGEN_REPLAY_COUNT=100 PKTGEN_REPLAY_TIME=60 SAMPLE=100 --cluster client=$CLIENT,nic=0 dut=$SERVER,nic=0 --graph-size 12 10 --single-output $RESULT_DIR/pmp.csv --cluster-autosave --result-path $RESULT_DIR/results --variables MINFREQ=$MIN_FREQ --config n_runs=20 --show-cmd | tee $RESULT_DIR/pmp-nosleep-$HOSTNAME.log

# # No need to test no_sleep again, already done in experiment 1
# echo "Experiment 3 : Extensive testing"
# RESULT_DIR=$MAIN_DIR/optimums/
# npf-compare.py "fastclick-sleepmodes+SLEEP_MODE=no_sleep,SLOTH_ENABLED=0,dpdk:Polling" "fastclick-sleepmodes+SLEEP_MODE=add_simple_intr,SLOTH_ENABLED=0,dpdk:Progressive nanosleeping + interrupt" --test  $SCRIPT_PATH --exp-design "allzlt(GEN_RATE,PPS)" --graph-filename $RESULT_DIR/pmp-$HOSTNAME.pdf --tags applications optimums dpdk $SETUP udpgen prate promisc  --variables MINFREQ=1000 DISTRIB=EXP SLO_LATENCY=50 LIMIT=50000000 PKTGEN_REPLAY_COUNT=100 PKTGEN_REPLAY_TIME=60 SAMPLE=100 --cluster client=$CLIENT,nic=0 dut=$SERVER,nic=0 --graph-size 12 10 --single-output $RESULT_DIR/pmp.csv --cluster-autosave --result-path $RESULT_DIR/cache/ --show-cmd --preserve-temporaries | tee $RESULT_DIR/pmp-nosleep-$HOSTNAME.log

# # Get data to find optimums
# echo "Experiment 5 : Finding optimums"
# RESULT_DIR=$MAIN_DIR/optimums/
# npf-compare.py "fastclick-sleepmodes+SLEEP_MODE=hrsleep_add_simple_intr,SLOTH_ENABLED=0,dpdk:Smarter sleeping + interrupt" --test  $SCRIPT_PATH --exp-design "allzlt(GEN_RATE,PPS)" --graph-filename $RESULT_DIR/pmp-$HOSTNAME.pdf --tags optimums dpdk $SETUP udpgen prate promisc  --variables MINFREQ=800 DISTRIB=EXP SLO_LATENCY=50 LIMIT=50000000 PKTGEN_REPLAY_COUNT=100 PKTGEN_REPLAY_TIME=60 SAMPLE=100 --cluster client=$CLIENT,nic=0 dut=$SERVER,nic=0 --graph-size 12 10 --single-output $RESULT_DIR/pmp.csv --cluster-autosave --result-path $RESULT_DIR/results --show-cmd  --preserve-temporaries | tee $RESULT_DIR/pmp-nosleep-$HOSTNAME.log

# Get data to find optimums
# echo "Experiment 5 : Finding optimums"
# RESULT_DIR=$MAIN_DIR/optimums/
# npf-compare.py "fastclick-sleepmodes+SLEEP_MODE=add_simple,SLOTH_ENABLED=0,dpdk:Smart progressive sleeping" --test  $SCRIPT_PATH --exp-design "allzlt(GEN_RATE,PPS)" --graph-filename $RESULT_DIR/pmp-$HOSTNAME.pdf --tags optimums dpdk $SETUP udpgen prate promisc  --variables MINFREQ=800 DISTRIB=EXP SLO_LATENCY=50 LIMIT=50000000 PKTGEN_REPLAY_COUNT=100 PKTGEN_REPLAY_TIME=60 SAMPLE=100 --cluster client=$CLIENT,nic=0 dut=$SERVER,nic=0 --graph-size 12 10 --single-output $RESULT_DIR/pmp.csv --cluster-autosave --result-path $RESULT_DIR/results --show-cmd  --preserve-temporaries | tee $RESULT_DIR/pmp-nosleep-$HOSTNAME.log


# # Run Sloth comparatively to other methods
# echo "Experiment 6 : Sloth vs others"
# RESULT_DIR=$MAIN_DIR/sloth-vs-others/
# npf-compare.py "fastclick-sleepmodes+SLEEP_MODE=sloth:Sloth" "fastclick-sleepmodes+SLEEP_MODE=no_sleep,SLOTH_ENABLED=0:Polling" "fastclick-sleepmodes+SLEEP_MODE=hr2sleep_add_simple:Progressive sleeping" "fastclick-sleepmodes+SLEEP_MODE=hr2sleep_constant_simple:Constant sleeping" "fastclick-sleepmodes+SLEEP_MODE=intr_hr2sleep_constant_power:Constant sleeping + interrupt" "fastclick-sleepmodes+SLEEP_MODE=hr2sleep_add_metronome:Metronome" "fastclick-sleepmodes+SLEEP_MODE=tupe:TUPE" --force-retest  --test  $SCRIPT_PATH --graph-filename $RESULT_DIR/pmp-$HOSTNAME.pdf --tags dpdk sloth $SETUP udpgen prate promisc --variables SLOTH_SOURCE_CSV=$OPTIMUMS_FILE NUMA_NODE=1 LIMIT=1000000  PKTGEN_REPLAY_COUNT=100 PKTGEN_REPLAY_TIME=60 SAMPLE=100 --cluster client=$CLIENT,nic=0 dut=$SERVER,nic=0 --graph-size 12 10 --single-output $RESULT_DIR/pmp.csv --cluster-autosave --result-path ./$RESULT_DIR/results/ --variables MINFREQ=1000  --show-cmd | tee $RESULT_DIR/pmp-nosleep-$HOSTNAME.log

# # Dynamic load
# echo "Experiment 7 : Dynamic load"
# RESULT_DIR=$MAIN_DIR/dynamic-load/
# npf-compare.py "fastclick-sleepmodes+SLEEP_MODE=sloth:Sloth" "fastclick-sleepmodes+SLEEP_MODE=no_sleep,SLOTH_ENABLED=0:Polling" "fastclick-sleepmodes+SLEEP_MODE=hr2sleep_add_simple:Progressive sleeping" "fastclick-sleepmodes+SLEEP_MODE=hr2sleep_constant_simple:Constant sleeping" "fastclick-sleepmodes+SLEEP_MODE=intr_hr2sleep_constant_power:Constant sleeping + interrupt" "fastclick-sleepmodes+SLEEP_MODE=hr2sleep_add_metronome:Metronome" "fastclick-sleepmodes+SLEEP_MODE=tupe:TUPE" --force-retest --test $SCRIPT_PATH --graph-filename $RESULT_DIR/pmp-$HOSTNAME.pdf --tags dpdk trace timing iterative $SETUP --variables SLOTH_SOURCE_CSV=$OPTIMUMS_FILE NUMA_NODE=1 LIMIT=1000000 PKTGEN_REPLAY_COUNT=100 PKTGEN_REPLAY_TIME=60 SAMPLE=100 --cluster client=$CLIENT,nic=0 dut=$SERVER,nic=0 --graph-size 12 10 --single-output $RESULT_DIR/pmp.csv --cluster-autosave --result-path $RESULT_DIR/results/ --variables MINFREQ=1000  --show-cmd | tee $RESULT_DIR/pmp-nosleep-$HOSTNAME.log

# echo "Experiment 8 : Different applications with static throughput"
# RESULT_DIR=$MAIN_DIR/apps-static/
# npf-compare.py "fastclick-sleepmodes+SLEEP_MODE=sloth:Sloth" "fastclick-sleepmodes+SLEEP_MODE=no_sleep:Polling" "fastclick-sleepmodes+SLEEP_MODE=tupe:TUPE" --force-retest --test  $SCRIPT_PATH --graph-filename $RESULT_DIR/pmp-$HOSTNAME.pdf --tags apps dpdk $SETUP udpgen prate promisc --variables SLOTH_SOURCE_CSV=$OPTIMUMS_FILE NUMA_NODE=1 LIMIT=1000000  PKTGEN_REPLAY_COUNT=100 PKTGEN_REPLAY_TIME=60 SAMPLE=100 --cluster client=$CLIENT,nic=0 dut=$SERVER,nic=0 --graph-size 12 10 --single-output $RESULT_DIR/pmp.csv --cluster-autosave --result-path ./$RESULT_DIR/results/ --variables MINFREQ=1000  --show-cmd | tee $RESULT_DIR/pmp-nosleep-$HOSTNAME.log

# # Run Sloth comparatively to other methods
# echo "Experiment 9 : Dynamic load for multiple applications"
# RESULT_DIR=$MAIN_DIR/apps-dynamic/
# npf-compare.py "fastclick-sleepmodes+SLEEP_MODE=sloth:Sloth" "fastclick-sleepmodes+SLEEP_MODE=no_sleep:Polling" "fastclick-sleepmodes+SLEEP_MODE=tupe:TUPE"  --test $SCRIPT_PATH --graph-filename $RESULT_DIR/pmp-$HOSTNAME.pdf --tags dpdk trace apps timing iterative $SETUP --variables SLOTH_SOURCE_CSV=$OPTIMUMS_FILE NUMA_NODE=1 LIMIT=1000000 PKTGEN_REPLAY_COUNT=100 PKTGEN_REPLAY_TIME=10 SAMPLE=100 --cluster client=$CLIENT,nic=0 dut=$SERVER,nic=0 --graph-size 12 10 --single-output $RESULT_DIR/pmp.csv --cluster-autosave --result-path $RESULT_DIR/results/ --variables MINFREQ=1000  --show-cmd | tee $RESULT_DIR/pmp-nosleep-$HOSTNAME.log
