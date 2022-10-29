set_host_options -max_cores 8
suppress_message ATTR-11
suppress_message NEX-030
suppress_message NEX-022
suppress_message NEX-028

open_lib ./impl/vga
open_block clock_final_opto

source /nethome/hhsiao30/rl_project/write_gnn_inputs.tcl
########################
# link technology file #
########################
source /nethome/ylu478/gtcad_snps/icc2_gui_user_pref.tcl
source /nethome/ylu478/capture_maps.tcl
source /nethome/ylu478/track_util_icc2.tcl

###################
# ROUTE_AUTO ACTIONS
###################
set action_path "route_auto_actions.txt"
set fp [open $action_path r]
while { [gets $fp line] >= 0 } {
    set s [split $line " "]
    set par [lindex $s 0]
    set val [lindex $s 1]
    set_app_options -name $par -value $val
    puts "set_app_options -name $par -value $val"
}
close $fp

route_auto
save_block -as route_auto -compress
report_qor > route_auto.qor
report_power -sig 5 > route_auto.power
report_global_timing > route_auto.timing
write_def -include { cells nets } route_auto.def
write_lef -design route_auto route_auto.lef
writeNodeFeatures route_auto

exit
