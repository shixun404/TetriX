set term post eps enh "Arial" 32 color
set output "testdata.txt_1.eps"
set datafile missing "-"
set key inside top right Left
#set nokey

set auto x
#set xtic 20
#set xrange [0:3600]
#set grid y

set style line 1 lt 1 lc rgb "blue" lw 7
set style line 2 lt 2 lc rgb "purple" lw 7
set style line 3 lt 3 lc rgb "green" lw 7
set style line 4 lt 4 lc rgb "yellow" lw 7
set style line 5 lt 5 lc rgb "cyan" lw 7
set style line 6 lt 6 lc rgb "red" lw 7


set xlabel "Data Values"
set ylabel "CDF"

set style fill solid border -1
#set style data fillsteps
set style data lines
#set size 3,1
#set xtic rotate by -45
plot 'cdf/testdata.txt_1.cdf' using 1:2 ti 'testdata #1' ls 1
