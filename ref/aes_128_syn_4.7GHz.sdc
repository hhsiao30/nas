###################################################################

# Created by write_sdc on Sun Nov 18 11:49:09 2018

###################################################################
set sdc_version 2.1

set_units -time ns -resistance kOhm -capacitance pF -voltage V -current mA
set_max_fanout 20 [get_ports clk]
set_max_fanout 20 [get_ports reset]
set_propagated_clock [get_ports clk]
create_clock [get_ports clk]  -period 0.2128  -waveform {0 0.1064}
set_false_path   -from [get_ports reset]
