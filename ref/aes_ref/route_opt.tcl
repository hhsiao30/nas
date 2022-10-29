set_host_options -max_cores 8
suppress_message ATTR-11
suppress_message NEX-030
suppress_message NEX-022
suppress_message NEX-028

open_lib ./impl/aes
open_block route_auto

source /nethome/hhsiao30/rl_project/write_gnn_inputs.tcl
########################
# link technology file #
########################
source /nethome/ylu478/gtcad_snps/icc2_gui_user_pref.tcl
source /nethome/ylu478/capture_maps.tcl
source /nethome/ylu478/track_util_icc2.tcl

###################
# ROUTE_OPT ACTIONS
###################
set action_path "route_opt_actions.txt"
puts $action_path
set fp [open $action_path r]
while { [gets $fp line] >= 0 } {
    set s [split $line " "]
    set par [lindex $s 0]
    set val [lindex $s 1]
    set_app_options -name $par -value $val
    puts "set_app_options -name $par -value $val"
}
close $fp

set_app_options -list { route_opt.flow.enable_ccd true }
route_opt
save_block -as route_opt3 -compress
report_qor > route_opt3.qor
report_power -sig 5 > route_opt3.power
report_global_timing > route_opt3.timing
write_def -include { cells nets } route_opt3.def
write_lef -design route_opt3 route_opt3.lef
writeNodeFeatures route_opt3

# set_app_options -list { route_opt.flow.enable_ccd true }
# route_opt
# save_block -as route_opt1 -compress
# report_qor > route_opt1.qor
# report_power -sig 5 > route_opt1.power
# report_global_timing > route_opt1.timing
# write_def -include { cells nets } route_opt1.def
# write_lef -design route_opt1 route_opt1.lef
# writeNodeFeatures route_opt1

# set_app_options -list { route_opt.flow.enable_ccd false }
# route_opt
# save_block -as route_opt2 -compress
# report_qor > route_opt2.qor
# report_power -sig 5 > route_opt2.power
# report_global_timing > route_opt2.timing
# write_def -include { cells nets } route_opt2.def
# write_lef -design route_opt2 route_opt2.lef
# writeNodeFeatures route_opt2
 
# set_app_options -list { route_opt.flow.size_only_mode equal_or_smaller }
# route_opt
# save_block -as route_opt3 -compress
# report_qor > route_opt3.qor
# report_power -sig 5 > route_opt3.power
# report_global_timing > route_opt3.timing
# write_def -include { cells nets } route_opt3.def
# write_lef -design route_opt3 route_opt3.lef
# writeNodeFeatures route_opt3

exit
