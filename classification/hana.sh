for w in 8 4 2
do
for a in 8 4 2
do
for d in distill true_data random_data
do
	
	#echo $w $a $d
	cmd="CUDA_VISIBLE_DEVICES='1' python uniform_test.py --weight-bit $w --activation-bit $a --datatype $d"
	echo $cmd | bash

done
done
done		
