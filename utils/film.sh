#!/bin/bash

scan() {
f=scans/scan_$i
last_x=$(tail -1 $f | awk '{print $2}')
last_y=$(tail -1 $f | awk '{print $3}')
first_frame=1
last_frame=$(wc $f | awk '{print $1}')

if [ "$last_frame" -gt 0 ]
then
        echo """
set terminal pngcairo size 800,400 font 'arial bold,14'
set output 'scans/scan_$i.png'
set origin 0,0
#set xzeroaxis
plot 'scans/scan_$i' u 2:3 w p pt 1
        """ | gnuplot
fi
}


for i in {1..100} 
do
	echo "faccio scan.png di $i"
	#/bin/bash scan.sh $i
	scan $i
done

ffmpeg -y -framerate 2 -i scans/scan_%d.png scans.mp4
totem scans.mp4

echo "finito, pace e bene"
