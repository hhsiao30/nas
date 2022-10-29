ref_sdc=$1
new_sdc=$2
period=$3
waveform=$4
echo $ref_sdc
echo $new_sdc
echo $period
echo $waveform
rm -f $new_sdc
content= `cat $ref_sdc`
echo $content
awk '{if ($1 == "create_clock") {$5= $period; $8= $waveform;} print $0}' $ref_sdc