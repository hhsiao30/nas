###################################################################

# Created by write_sdc on Sat Nov 17 22:08:18 2018

###################################################################
set sdc_version 1.9

set_units -time ns -resistance kOhm -capacitance pF -voltage V -current mA
set_max_fanout 20 [get_ports reset]
set_max_fanout 20 [get_ports clock]
set_propagated_clock [get_ports clock]
create_clock [get_ports clock]  -period 0.5  -waveform {0 0.250}
set_false_path   -from [get_ports reset]
