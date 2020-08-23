for w in 8 4 2
do
for a in 8 4 2
do
for d in distill true_data random_data
do
for s in 1 10 100 1000 6000
do
	
	echo "EVAL" $w $a $d $s

	cmd="CUDA_VISIBLE_DEVICES='1' python uniform_test.py --weight-bit $w --activation-bit $a --datatype $d --distill-size $s"
	echo $cmd | bash

done
done
done
done		
