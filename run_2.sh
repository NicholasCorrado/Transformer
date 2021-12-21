#!/bin/bash
declare -a stock_1s=("APA" "CVX" "MRO" "XOM")
declare -a stock_2s=("AAPL" "ADBE" "GOOGL" "MSI")


for i in 1 2 3
do
    mkdir results_$i
    
    for s1 in "${stock_1s[@]}"
    do
        mkdir results
        # APA  = APA Corp. (Oil & Gas Exploration & Production)
        # AAPL = Apple (Technology Hardware, Storage & Peripherals)
        python driver_2.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 $s1
        python driver_2.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 $s1
        python driver_2.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 $s1
        python driver_2.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 $s1
        mv results results_$s1 -f
        mv results_$s1 results_$i/ -f
    done
done


# for s1 in "${stock_1s[@]}"
# do
#     mkdir results 
#     # APA  = APA Corp. (Oil & Gas Exploration & Production)
#     # AAPL = Apple (Technology Hardware, Storage & Peripherals)
#     python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 2 --stock-name-1 $s1 --stock-name-2 AAPL --energy-to-tech 0
#     python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 2 --stock-name-1 $s1 --stock-name-2 AAPL --energy-to-tech 0
#     python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 2 --stock-name-1 $s1 --stock-name-2 AAPL --energy-to-tech 0
#     python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 2 --stock-name-1 $s1 --stock-name-2 AAPL --energy-to-tech 0
#     mv results results_AAPL_$s1
# done

# mkdir results 
# # APA  = APA Corp. (Oil & Gas Exploration & Production)
# # AAPL = Apple (Technology Hardware, Storage & Peripherals)
# python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL --energy-to-tech 0
# python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL --energy-to-tech 0
# python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL --energy-to-tech 0
# python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL --energy-to-tech 0
# mv results results_AAPL_APA

# mkdir results 
# # CVX = Chevron (Integrated Oil & Gas)
# # AAPL = Apple (Technology Hardware, Storage & Peripherals)
# python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL --energy-to-tech 0
# python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL --energy-to-tech 0
# python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL --energy-to-tech 0
# python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL --energy-to-tech 0
# mv results results_AAPL_CVX


# mkdir results 
# # XOM = ExxonMobil (Integrated Oil & Gas)
# # AAPL = Apple (Technology Hardware, Storage & Peripherals)
# python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL --energy-to-tech 0
# python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL --energy-to-tech 0
# python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL --energy-to-tech 0
# python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL --energy-to-tech 0
# mv results results_AAPL_MRO

