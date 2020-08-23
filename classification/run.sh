for MODEL in resnet18 
do
	echo Testing $MODEL ...
	python uniform_test.py 		\
		--dataset=imagenet 		\
		--model=$MODEL 			\
		--batch_size=8 		\
		--test_batch_size=8
done
