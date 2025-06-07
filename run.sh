# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES='1'

# 定义固定参数
hop=2
max_nodes_per_hop=100
hv_num_list=(500) # 1000)
celltype_list=(mHSC-L) #(hESC hHEP mDC mHSC-E mHSC-GM mHSC-L)
net_list=(Non-Specific) # Non-Specific STRING Lofgof)

for net in "${net_list[@]}"; do
    for hv_num in "${hv_num_list[@]}"; do
        for celltype in "${celltype_list[@]}"; do
            python Main.py \
                            --HV-num $hv_num \
                            --celltype $celltype \
                            --hop $hop \
                            --max-nodes-per-hop $max_nodes_per_hop \
                            --net $net \
                            # --fold True \

        done                
    done
done


# noise_list=(0 0.05 0.1 0.15 0.2 0.25 0.3)

# for noise in "${noise_list[@]}"; do 
#     python Main.py \
#                     --HV-num 500 \
#                     --celltype mHSC-L \
#                     --hop $hop \
#                     --max-nodes-per-hop $max_nodes_per_hop \
#                     --net STRING \
#                     --fold True \
#                     --noise $noise \

# done
                    