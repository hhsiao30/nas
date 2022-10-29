###################################################################

# Created by write_sdc on Mon Nov 11 12:26:27 2019

###################################################################
set sdc_version 2.1

set_units -time ps -resistance kOhm -capacitance fF -voltage V -current mA
set_max_fanout 20 [current_design]
create_clock [get_ports ispd_clk]  -name clock  -period 0.526  -waveform {0 0.263}
