#!/bin/sh
for i in {1..10}
do
    python3 -m dataset.tentative --sample_id=$i > "../data_christmas/out_$i.txt" &
done

#  run.sh
#  
#
#  Created by KiJung on 7/13/17.
#  
#  https://stackoverflow.com/questions/35869137/avoid-tensorflow-print-on-standard-error
