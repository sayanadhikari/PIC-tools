#!/bin/bash

./disprel.py -i ../bounded_sys_len/1D_bounded_Tb_05_L_double/Ex/ -per -pl -n omega_pi
echo "1/4 done" 
./disprel.py -i ../bounded_sys_len/1D_bounded_Tb_10_L_double/Ex/ -per -pl -n omega_pi
echo "2/4 done"	
./disprel.py -i ../bounded_sys_len/1D_bounded_Tb_75_L_half/Ex/ -per -pl -n omega_pi
echo "3/4 done"	
./disprel.py -i ../bounded_sys_len/1D_bounded_Tb_75_L_double/Ex/ -per -pl -n omega_pi
echo "4/4 done"	
