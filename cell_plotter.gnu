N = 64  #Number of cells in simulation
d = sqrt(2/sqrt(3))
h = sqrt(3)*d/2

start = 0 #Start file number
stop = 1000 #End file number

set terminal pngcairo rounded size 800,800
set style line 1 lt 3 lc rgb 'black' pt 7 lw 2

set xrange [0:d*sqrt(N)]
set yrange [0:h*sqrt(N)]

point_files(n) = sprintf('./PointFiles/points%06d.txt', n)
cell_files(n) = sprintf('./CellFiles/cells%06d.gnu', n)

do for [i = start:stop:10] {
    outfile = sprintf('./Figures/figures%06d.png',i)
    set output outfile
    unset key
    plot  cell_files(i) w l ls 1

}
