set font "arial, 24"
set style line 1 lt 1 lc rgb "#f20c0c" lw 1 pt 7 ps 1
set title "Lattice Distance vs Polykov" font "arial, 12"
set xlabel "Distance on Lattice" font "arial, 12"
set ylabel "Polykov Loop" font "arial, 12"
plot 'DistvsPolykov.dat' using 1:2:3 w yerr ls 1 title 'Size 64'
pause -1 "Hit any key to continue"
