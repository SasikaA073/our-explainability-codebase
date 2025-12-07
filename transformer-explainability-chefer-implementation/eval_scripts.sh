# ViT Explainability

# Available methods:
# - transformer_attribution
# - rollout
# - lrp
# - full_lrp
# - v_gradcam
# - lrp_last_layer
# - lrp_second_layer
# - gradcam


# Segmentation Evaluation
# ViT-Base
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/imagenet_seg_eval.py \
#     --method transformer_attribution \
#     --imagenet-seg-path imagenet/segmentation/gtsegs_ijcv.mat \
#     --num-samples 5 \
#     --model-name vit-base > logs/vit-base_transformer_attribution_1_sample.log 2>&1 

# ViT-Large
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/imagenet_seg_eval.py \
    --method transformer_attribution \
    --imagenet-seg-path imagenet/segmentation/gtsegs_ijcv.mat \
    --num-samples 5 \
    --model-name vit-large > logs/vit-large_transformer_attribution_5_sample.log 2>&1 


# # Perturbation Evaluation
# ## Step 1: Generate Visualizations 
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/generate_visualizations.py \
#   --method transformer_attribution \
#   --imagenet-validation-path /path/to/imagenet_validation_directory \
#   --vis-cls top

# ## Step 2: Run Perturbation Test 
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH python3 baselines/ViT/pertubation_eval_from_hdf5.py \
#   --method transformer_attribution \
#   --neg
