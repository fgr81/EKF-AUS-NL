#!/bin/bash

f=../trajectory.dat
last_x=$(tail -1 $f | awk '{print $1}')
last_y=$(tail -1 $f | awk '{print $2}')
last_phi=$(tail -1 $f | awk '{print $3}')
first_frame=1
last_frame=$(wc $f | awk '{print $1}')
echo "ciao $last_x $last_y $last_phi  $first_frame $last_frame"

echo """
set terminal pngcairo size 800,400 font 'arial bold,14'
set label at screen 0.05,0.9 tc lt 7 \"Last position=$last_x $last_y $last_phi\"
set output 'trajectory.png'
set xrange ['$firts_frame' : '$last_frame']
set xlabel  'Frame'
set xtics 21600
plot '../trajectory.dat' u 1:2 w l lt 7 lw 2 t''
""" | gnuplot

