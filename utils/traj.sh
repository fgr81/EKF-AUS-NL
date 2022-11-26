#!/bin/bash

f=../trajectory.dat
last_x=$(tail -1 $f | awk '{print $1}')
last_y=$(tail -1 $f | awk '{print $2}')
last_phi=$(tail -1 $f | awk '{print $3}')
first_frame=1
last_frame=$(wc $f | awk '{print $1}')

echo """
set terminal pngcairo size 800,400 font 'arial bold,14'
set output 'trajectory.png'
set xtics autofreq
plot '../trajectory.dat' u 1:2 w p pt 7, '../percorso.dat' u 1:2 w p pt 6  
""" | gnuplot

display trajectory.png
