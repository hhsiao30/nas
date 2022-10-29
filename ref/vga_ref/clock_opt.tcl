set_host_options -max_cores 8
suppress_message ATTR-11
suppress_message NEX-030
suppress_message NEX-022
suppress_message NEX-028

open_lib ./impl/vga
open_block final_opto
read_sdc ./vga.sdc

source /nethome/hhsiao30/rl_project/write_gnn_inputs.tcl
########################
# link technology file #
########################
source /nethome/ylu478/gtcad_snps/icc2_gui_user_pref.tcl
source /nethome/ylu478/capture_maps.tcl
source /nethome/ylu478/track_util_icc2.tcl

###################
# CLOCK_OPT ACTIONS
###################
set action_path "clock_opt_actions.txt"
set fp [open $action_path r]
while { [gets $fp line] >= 0 } {
    set s [split $line " "]
    set par [lindex $s 0]
    set val [lindex $s 1]
    set_app_options -name $par -value $val
    puts "set_app_options -name $par -value $val"
}
close $fp

clock_opt -from build_clock -to build_clock
save_block -as build_clock -compress
report_qor > build_clock.qor
report_power -sig 5 > build_clock.power
report_global_timing > build_clock.timing
write_def -include { cells nets } build_clock.def
write_lef -design build_clock build_clock.lef
writeNodeFeatures build_clock

clock_opt -from route_clock -to route_clock
save_block -as route_clock -compress
report_qor > route_clock.qor
report_power -sig 5 > route_clock.power
report_global_timing > route_clock.timing
write_def -include { cells nets } route_clock.def
write_lef -design route_clock route_clock.lef
writeNodeFeatures route_clock

clock_opt -from final_opto -to final_opto
save_block -as clock_final_opto -compress
report_qor > clock_final_opto.qor
report_power -sig 5 > clock_final_opto.power
report_global_timing > clock_final_opto.timing
write_def -include { cells nets } clock_final_opto.def
write_lef -design clock_final_opto clock_final_opto.lef
writeNodeFeatures clock_final_opto

exit
