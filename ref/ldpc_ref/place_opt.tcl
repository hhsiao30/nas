set_host_options -max_cores 8
suppress_message ATTR-11
suppress_message NEX-030
suppress_message NEX-022i
suppress_message NEX-028

open_lib ./impl/ldpc
open_block initial_opto

read_sdc ./ldpc.sdc

source /nethome/hhsiao30/rl_project/write_gnn_inputs.tcl
########################
# link technology file #
########################
source /nethome/ylu478/gtcad_snps/icc2_gui_user_pref.tcl
source /nethome/ylu478/capture_maps.tcl
source /nethome/ylu478/track_util_icc2.tcl

###################
# PLACEMENT ACTIONS
###################
set action_path "place_opt_actions.txt"
set fp [open $action_path r]
while { [gets $fp line] >= 0 } {
    set s [split $line " "]
    set par [lindex $s 0]
    set val [lindex $s 1]
    set_app_options -name $par -value $val
    puts "set_app_options -name $par -value $val"
}
close $fp

place_opt -from final_opto -to final_opto
save_block -as final_opto -compress
report_qor > final_opto.qor
report_power -sig 5 > final_opto.power
report_global_timing > final_opto.timing
write_def -include { cells nets } final_opto.def
write_lef -design final_opto final_opto.lef
writeNodeFeatures final_opto

place_pins -self

exit
