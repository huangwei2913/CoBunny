for i in {0..7}; do
	    python -m bunny.eval.model_vqa_loader_mme_mixencoders \
		            --model-path /mnt/CoBunny/checkpoints-finetune/phi-1.5-bunny-mixed-final \
			            --image-folder ./eval/mme/MME_Benchmark_release_version/MME_Benchmark \
				            --question-file ./eval/mme/bunny_mme.jsonl \
					            --answers-file ./eval/mme/answers/part_${i}.jsonl \
						            --num-chunks 8 \
							            --chunk-idx $i \
								            --temperature 0 \
									            --conv-mode bunny & 
									    done
									    wait
