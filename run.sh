#!/bin/bash

mkdir results 
# APA  = APA Corp. (Oil & Gas Exploration & Production)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL 
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL 
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL 
mv results results_APA_AAPL

mkdir results 
# CVX = Chevron (Integrated Oil & Gas)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL 
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL 
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL 
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL 
mv results results_CVX_AAPL


mkdir results 
# XOM = ExxonMobil (Integrated Oil & Gas)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL 
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL 
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL 
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL 
mv results results_MRO_AAPL



mkdir results 
# APA  = APA Corp. (Oil & Gas Exploration & Production)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL --energy-to-tech 0
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL --energy-to-tech 0
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL --energy-to-tech 0
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 APA --stock-name-2 AAPL --energy-to-tech 0
mv results results_AAPL_APA

mkdir results 
# CVX = Chevron (Integrated Oil & Gas)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL --energy-to-tech 0
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL --energy-to-tech 0
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL --energy-to-tech 0
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 CVX --stock-name-2 AAPL --energy-to-tech 0
mv results results_AAPL_CVX


mkdir results 
# XOM = ExxonMobil (Integrated Oil & Gas)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL --energy-to-tech 0
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL --energy-to-tech 0
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL --energy-to-tech 0
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 MRO --stock-name-2 AAPL --energy-to-tech 0
mv results results_AAPL_MRO





mkdir results 
# XOM = ExxonMobil (Integrated Oil & Gas)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 GOOGL 
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 GOOGL
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 GOOGL
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 GOOGL
mv results results_XOM_GOOGL


mkdir results 
# XOM = ExxonMobil (Integrated Oil & Gas)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 ADBE 
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 ADBE
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 ADBE
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 ADBE
mv results results_XOM_ADBE

mkdir results 
# XOM = ExxonMobil (Integrated Oil & Gas)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 MSI
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 MSI
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 MSI
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 MSI
mv results results_XOM_MSI





mkdir results 
# XOM = ExxonMobil (Integrated Oil & Gas)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 GOOGL --energy-to-tech 0
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 GOOGL --energy-to-tech 0
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 GOOGL --energy-to-tech 0
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 GOOGL --energy-to-tech 0
mv results results_GOOGL_XOM


mkdir results 
# XOM = ExxonMobil (Integrated Oil & Gas)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 ADBE --energy-to-tech 0
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 ADBE --energy-to-tech 0
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 ADBE --energy-to-tech 0
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 ADBE --energy-to-tech 0
mv results results_ADBE_XOM

mkdir results 
# XOM = ExxonMobil (Integrated Oil & Gas)
# AAPL = Apple (Technology Hardware, Storage & Peripherals)
python driver.py --s 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 MSI --energy-to-tech 0
python driver.py --s 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 MSI --energy-to-tech 0
python driver.py --l 1 --a 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 MSI --energy-to-tech 0
python driver.py --l 1 --c 1 --vocab-size 10000 --batch-size 64 --num-epochs 100 --stock-name-1 XOM --stock-name-2 MSI --energy-to-tech 0
mv results results_MSI_XOM

